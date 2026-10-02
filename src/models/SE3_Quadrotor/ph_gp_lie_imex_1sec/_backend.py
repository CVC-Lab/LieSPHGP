"""Private fixed-feature setup and variational-parameter backend."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .gp_model import GP_Model

Array = jax.Array
GP_NAMES = ("M1", "M2", "Dv", "Dw", "V", "g")
OUTPUT_DIMENSIONS = {"M1": 1, "M2": 6, "Dv": 6, "Dw": 6, "V": 1, "g": 24}
POSTERIOR_INITIAL_LOG_STD = -5.0
MASS_EIGENVALUE_FLOOR = 1.0e-2
MASS_RESIDUAL_FRACTION = 0.10
DISSIPATION_RESIDUAL_FRACTION = 0.25
CONTROL_RESIDUAL_SCALE = 0.10
LENGTH_SCALE_FLOOR = 5.0e-2
def _student_t_frequencies(
    key: Array,
    count: int,
    dimension: int,
    *,
    nu: float,
) -> Array:
    """Sample unit-lengthscale Matérn random-feature frequencies."""
    key_normal, key_gamma = jax.random.split(key)
    degrees = 2.0 * nu
    gamma = jax.random.gamma(key_gamma, degrees / 2.0, shape=(count,)) / 0.5
    normal = jax.random.normal(key_normal, (count, dimension))
    return normal / jnp.sqrt(gamma[:, None] / degrees)


def initialize_euclidean_features(
    key: Array,
    dimension: int,
    features: int,
    *,
    nu: float = 2.5,
    center: Array | np.ndarray | None = None,
    standard_deviation: Array | np.ndarray | None = None,
) -> dict[str, Array]:
    """Initialize a normalized Euclidean Matérn feature setup."""
    key_frequency, key_phase = jax.random.split(key)
    if center is None:
        center = np.zeros(dimension, dtype=np.float32)
    if standard_deviation is None:
        standard_deviation = np.ones(dimension, dtype=np.float32)
    scale = np.maximum(np.asarray(standard_deviation, dtype=np.float32), 1.0e-3)
    return {
        "frequency": _student_t_frequencies(
            key_frequency, features, dimension, nu=nu
        ),
        "phase": jax.random.uniform(key_phase, (features,)) * (2.0 * jnp.pi),
        "feature_scale": jnp.asarray((2.0 / features) ** 0.5, dtype=jnp.float32),
        "center": jnp.asarray(center, dtype=jnp.float32),
        "standard_deviation": jnp.asarray(scale, dtype=jnp.float32),
    }


def _symmetric_traceless_basis(degree: int) -> np.ndarray:
    """Return an orthonormal basis for rank-degree harmonic tensors in R3."""
    if degree == 0:
        return np.ones((1,), dtype=np.float32)
    size = 3**degree
    constraints: list[np.ndarray] = []
    for indices in np.ndindex(*(3,) * degree):
        flat = np.ravel_multi_index(indices, (3,) * degree)
        for axis in range(degree - 1):
            swapped = list(indices)
            swapped[axis], swapped[axis + 1] = swapped[axis + 1], swapped[axis]
            other = np.ravel_multi_index(tuple(swapped), (3,) * degree)
            if flat < other:
                row = np.zeros(size, dtype=np.float64)
                row[flat], row[other] = 1.0, -1.0
                constraints.append(row)
    if degree >= 2:
        for remainder in np.ndindex(*(3,) * (degree - 2)):
            row = np.zeros(size, dtype=np.float64)
            for coordinate in range(3):
                indices = (coordinate, coordinate, *remainder)
                row[np.ravel_multi_index(indices, (3,) * degree)] = 1.0
            constraints.append(row)
    matrix = np.stack(constraints) if constraints else np.zeros((0, size))
    _, singular_values, right = np.linalg.svd(matrix, full_matrices=True)
    tolerance = max(matrix.shape, default=1) * np.finfo(np.float64).eps
    tolerance *= singular_values[0] if singular_values.size else 1.0
    rank = int(np.sum(singular_values > tolerance))
    basis = right[rank:].T
    expected = 2 * degree + 1
    if basis.shape[1] != expected:
        raise RuntimeError(
            f"SO(3) degree-{degree} basis has {basis.shape[1]} columns, expected {expected}"
        )
    return basis.T.reshape((expected,) + (3,) * degree).astype(np.float32)


def _maximum_degree(feature_budget: int) -> int:
    total = 0
    degree = 0
    while total + (2 * degree + 1) ** 2 <= feature_budget:
        total += (2 * degree + 1) ** 2
        degree += 1
    if degree == 0:
        raise ValueError("SO(3) feature budget must be at least 1")
    return degree - 1


def initialize_so3_features(
    key: Array,
    features: int,
    *,
    nu: float = 2.5,
) -> dict[str, Any]:
    """Initialize truncated Laplace--Beltrami Matérn features on SO(3)."""
    del key
    maximum_degree = _maximum_degree(features)
    bases = tuple(
        jnp.asarray(_symmetric_traceless_basis(degree))
        for degree in range(maximum_degree + 1)
    )
    return {
        "bases": bases,
        "laplacian_eigenvalues": jnp.asarray(
            [degree * (degree + 1) for degree in range(maximum_degree + 1)],
            dtype=jnp.float32,
        ),
        "representation_dimensions": jnp.asarray(
            [2 * degree + 1 for degree in range(maximum_degree + 1)],
            dtype=jnp.float32,
        ),
        "nu": jnp.asarray(nu, dtype=jnp.float32),
    }


def initialize_se3_features(
    key: Array,
    position_features: int,
    rotation_features: int,
    *,
    nu: float = 2.5,
    position_center: Array | np.ndarray | None = None,
    position_standard_deviation: Array | np.ndarray | None = None,
) -> dict[str, Any]:
    key_position, key_rotation = jax.random.split(key)
    return {
        "position": initialize_euclidean_features(
            key_position,
            3,
            position_features,
            nu=nu,
            center=position_center,
            standard_deviation=position_standard_deviation,
        ),
        "rotation": initialize_so3_features(key_rotation, rotation_features, nu=nu),
    }


def positive_length_scale(raw: Array) -> Array:
    return jax.nn.softplus(raw) + LENGTH_SCALE_FLOOR


def raw_length_scale(initial_value: float | Array) -> Array:
    value = jnp.asarray(initial_value, dtype=jnp.float32) - LENGTH_SCALE_FLOOR
    if bool(jnp.any(value <= 0.0)):
        raise ValueError(f"lengthscale must exceed {LENGTH_SCALE_FLOOR}")
    return jnp.log(jnp.expm1(value))


def euclidean_feature_dimension(feature_setup: dict[str, Array]) -> int:
    return int(feature_setup["phase"].shape[0])


def so3_feature_dimension(feature_setup: dict[str, Any]) -> int:
    return sum(int(basis.shape[0]) ** 2 for basis in feature_setup["bases"])


def euclidean_features(
    feature_setup: dict[str, Array], value: Array, length_scale: Array
) -> Array:
    normalized = (value - feature_setup["center"]) / feature_setup["standard_deviation"]
    argument = (
        feature_setup["frequency"] @ (normalized / length_scale)
        + feature_setup["phase"]
    )
    return feature_setup["feature_scale"] * jnp.cos(argument)


def _representation_matrix(rotation: Array, basis: Array) -> Array:
    degree = basis.ndim - 1
    if degree == 0:
        return jnp.ones((1, 1), dtype=rotation.dtype)
    output_indices = "abcd"[:degree]
    input_indices = "pqrs"[:degree]
    rotation_terms = ",".join(
        f"{output}{source}" for output, source in zip(output_indices, input_indices)
    )
    expression = f"{rotation_terms},k{input_indices}->k{output_indices}"
    rotated = jnp.einsum(expression, *([rotation] * degree), basis)
    return basis.reshape(basis.shape[0], -1) @ rotated.reshape(basis.shape[0], -1).T


def so3_features(
    feature_setup: dict[str, Any], rotation_flat: Array, length_scale: Array
) -> Array:
    rotation = rotation_flat.reshape(3, 3)
    nu = feature_setup["nu"]
    kappa_squared = 2.0 * nu / jnp.square(length_scale)
    coefficients = jnp.power(
        kappa_squared + feature_setup["laplacian_eigenvalues"], -(nu + 1.5)
    )
    normalization = jnp.sum(
        coefficients * feature_setup["representation_dimensions"]
    )
    feature_parts = []
    for index, basis in enumerate(feature_setup["bases"]):
        representation = _representation_matrix(rotation, basis)
        feature_parts.append(
            jnp.sqrt(coefficients[index] / normalization) * representation.reshape(-1)
        )
    return jnp.concatenate(feature_parts)


def product_features(left: Array, right: Array) -> Array:
    return jnp.outer(left, right).reshape(-1)


def se3_features(
    feature_setup: dict[str, Any], pose: Array, length_scales: Array
) -> Array:
    position_feature = euclidean_features(
        feature_setup["position"], pose[:3], length_scales[0]
    )
    rotation_feature = so3_features(
        feature_setup["rotation"], pose[3:12], length_scales[1]
    )
    return product_features(position_feature, rotation_feature)

DEFAULT_PHYSICS = {
    "mass": 0.027,
    "inertia_diagonal": (2.3951e-5, 2.3951e-5, 3.2347e-5),
    "gravity": 9.8,
    "damping_coefficient": 0.5,
}


def feature_normalization(states: np.ndarray) -> dict[str, np.ndarray]:
    """Compute training-only Euclidean normalization without touching controls."""
    flat = np.asarray(states, dtype=np.float32).reshape(-1, states.shape[-1])
    return {
        "position_center": flat[:, :3].mean(axis=0),
        "position_standard_deviation": np.maximum(flat[:, :3].std(axis=0), 1.0e-3),
        "velocity_center": flat[:, 12:15].mean(axis=0),
        "velocity_standard_deviation": np.maximum(flat[:, 12:15].std(axis=0), 1.0e-3),
        "omega_center": flat[:, 15:18].mean(axis=0),
        "omega_standard_deviation": np.maximum(flat[:, 15:18].std(axis=0), 1.0e-3),
    }


def physics_from_settings(settings: Mapping[str, Any]) -> dict[str, float | np.ndarray]:
    vehicle = settings.get("vehicle_parameters", {})
    inertia = np.asarray(
        vehicle.get("inertia", np.diag(DEFAULT_PHYSICS["inertia_diagonal"])),
        dtype=np.float32,
    )
    if inertia.shape != (3, 3) or not np.allclose(inertia, np.diag(np.diag(inertia))):
        raise ValueError("PH-GP physical prior currently requires diagonal body inertia")
    linear = float(settings.get("linear_damping_coefficient", DEFAULT_PHYSICS["damping_coefficient"]))
    angular = float(settings.get("angular_damping_coefficient", linear))
    if not np.isclose(linear, angular):
        raise ValueError("PH-GP physical prior currently requires equal linear/angular damping coefficients")
    return {
        "mass": float(vehicle.get("mass", DEFAULT_PHYSICS["mass"])),
        "inertia_diagonal": np.diag(inertia),
        "gravity": float(vehicle.get("gravity_acceleration", DEFAULT_PHYSICS["gravity"])),
        "damping_coefficient": linear,
    }


def _residual_factor_matrix(
    raw: Array,
    base_factor_diagonal: Array,
    *,
    residual_fraction: float,
    eigenvalue_floor: float,
) -> Array:
    diagonal = base_factor_diagonal * jnp.exp(residual_fraction * raw[:3])
    geometric = jnp.sqrt(
        jnp.asarray(
            [
                base_factor_diagonal[1] * base_factor_diagonal[0],
                base_factor_diagonal[2] * base_factor_diagonal[0],
                base_factor_diagonal[2] * base_factor_diagonal[1],
            ]
        )
    )
    off_diagonal = residual_fraction * geometric * raw[3:]
    lower = jnp.asarray(
        [
            [diagonal[0], 0.0, 0.0],
            [off_diagonal[0], diagonal[1], 0.0],
            [off_diagonal[1], off_diagonal[2], diagonal[2]],
        ],
        dtype=raw.dtype,
    )
    return lower @ lower.T + eigenvalue_floor * jnp.eye(3, dtype=raw.dtype)


def build_gp_setup(
    key: Array,
    *,
    normalization: Mapping[str, np.ndarray] | None = None,
    physics: Mapping[str, Any] | None = None,
    mass_1_feature_count: int = 220,
    mass_2_feature_budget: int = 220,
    dissipation_v_feature_count: int = 225,
    dissipation_w_feature_count: int = 220,
    potential_position_feature_count: int = 20,
    potential_rotation_feature_budget: int = 11,
    potential_include_rotation: bool = True,
    control_position_feature_count: int = 20,
    control_rotation_feature_budget: int = 11,
    direct_control_map: bool = False,
    matern_smoothness: float = 2.5,
    mass_eigenvalue_floor: float = MASS_EIGENVALUE_FLOOR,
    dissipation_enabled: bool = True,
    gp_model_backend: str | None = None,
    matern_length_scale: float | None = None,
    period: float | None = None,
    periodic_length_scale: float | None = None,
    periodic_harmonics: int | None = None,
    periodic_dimension: int | None = None,
) -> dict[str, Any]:
    """Build the fixed GP features, physics values, and model options."""
    normalization = dict(normalization or {})
    physics = dict(DEFAULT_PHYSICS if physics is None else physics)
    zero = np.zeros(3, dtype=np.float32)
    one = np.ones(3, dtype=np.float32)
    x_center = normalization.get("position_center", zero)
    x_scale = normalization.get("position_standard_deviation", one)
    v_center = normalization.get("velocity_center", zero)
    v_scale = normalization.get("velocity_standard_deviation", one)
    w_center = normalization.get("omega_center", zero)
    w_scale = normalization.get("omega_standard_deviation", one)

    if gp_model_backend is not None:
        if gp_model_backend != "utils-gp-model":
            raise ValueError("gp.model_backend must be null or utils-gp-model")
        exact_values = {
            "matern_length_scale": matern_length_scale,
            "period": period,
            "periodic_length_scale": periodic_length_scale,
            "periodic_harmonics": periodic_harmonics,
            "periodic_dimension": periodic_dimension,
        }
        missing = [name for name, value in exact_values.items() if value is None]
        if missing:
            raise ValueError(
                "The utils-gp-model backend requires: " + ", ".join(missing)
            )
        if not 0 <= int(periodic_dimension) < 9:
            raise ValueError("gp.periodic_dimension must be between 0 and 8")
        specifications = {
            "M1": (12, 1, mass_1_feature_count),
            "M2": (9, 6, mass_2_feature_budget),
            "Dv": (12, 6, dissipation_v_feature_count),
            "Dw": (12, 6, dissipation_w_feature_count),
            "V": (
                12,
                1,
                max(
                    potential_position_feature_count,
                    potential_rotation_feature_budget,
                ),
            ),
            "g": (
                12,
                24,
                max(control_position_feature_count, control_rotation_feature_budget),
            ),
        }
        gp_setup = {
            "_exact_specs": specifications,
            "_exact_settings": {
                "nu": float(matern_smoothness),
                "ell_m": float(matern_length_scale),
                "period": float(period),
                "ell_p": float(periodic_length_scale),
                "m_max": int(periodic_harmonics),
                "periodic_dim": int(periodic_dimension),
            },
            "_physics": {
                "mass": jnp.asarray(physics["mass"], dtype=jnp.float32),
                "inertia_diagonal": jnp.asarray(
                    physics["inertia_diagonal"], dtype=jnp.float32
                ),
                "gravity": jnp.asarray(physics["gravity"], dtype=jnp.float32),
                "damping_coefficient": jnp.asarray(
                    physics["damping_coefficient"], dtype=jnp.float32
                ),
                "mass_eigenvalue_floor": jnp.asarray(
                    mass_eigenvalue_floor, dtype=jnp.float32
                ),
                "dissipation_enabled": jnp.asarray(
                    float(dissipation_enabled), dtype=jnp.float32
                ),
            },
        }
        if not potential_include_rotation:
            gp_setup["_potential_position_only"] = jnp.asarray(True)
        if direct_control_map:
            gp_setup["_direct_control_map"] = jnp.asarray(True)
        return gp_setup

    keys = jax.random.split(key, 7)
    gp_setup = {
        "M1": initialize_euclidean_features(
            keys[0], 3, mass_1_feature_count,
            nu=matern_smoothness,
            center=x_center, standard_deviation=x_scale,
        ),
        "M2": initialize_so3_features(
            keys[1], mass_2_feature_budget, nu=matern_smoothness
        ),
        "Dv": initialize_euclidean_features(
            keys[2], 3, dissipation_v_feature_count,
            nu=matern_smoothness,
            center=v_center, standard_deviation=v_scale,
        ),
        "Dw": initialize_euclidean_features(
            keys[3], 3, dissipation_w_feature_count,
            nu=matern_smoothness,
            center=w_center, standard_deviation=w_scale,
        ),
        "V": (
            initialize_se3_features(
                keys[4], potential_position_feature_count,
                potential_rotation_feature_budget,
                nu=matern_smoothness,
                position_center=x_center,
                position_standard_deviation=x_scale,
            )
            if potential_include_rotation
            else initialize_euclidean_features(
                keys[4], 3, potential_position_feature_count,
                nu=matern_smoothness,
                center=x_center, standard_deviation=x_scale,
            )
        ),
        "g": initialize_se3_features(
            keys[5], control_position_feature_count,
            control_rotation_feature_budget,
            nu=matern_smoothness,
            position_center=x_center,
            position_standard_deviation=x_scale,
        ),
        "_physics": {
            "mass": jnp.asarray(physics["mass"], dtype=jnp.float32),
            "inertia_diagonal": jnp.asarray(
                physics["inertia_diagonal"], dtype=jnp.float32
            ),
            "gravity": jnp.asarray(physics["gravity"], dtype=jnp.float32),
            "damping_coefficient": jnp.asarray(
                physics["damping_coefficient"], dtype=jnp.float32
            ),
            "mass_eigenvalue_floor": jnp.asarray(
                mass_eigenvalue_floor, dtype=jnp.float32
            ),
            "dissipation_enabled": jnp.asarray(
                float(dissipation_enabled), dtype=jnp.float32
            ),
        },
    }
    if direct_control_map:
        # The key's presence is a static architecture marker inside JAX traces.
        gp_setup["_direct_control_map"] = jnp.asarray(True)
    return gp_setup


def feature_dimensions(gp_setup: Mapping[str, Any]) -> dict[str, int]:
    if "_exact_models" in gp_setup:
        return {
            name: int(model.Dm * model.Dp)
            for name, model in gp_setup["_exact_models"].items()
        }
    potential_uses_rotation = "position" in gp_setup["V"]
    return {
        "M1": euclidean_feature_dimension(gp_setup["M1"]),
        "M2": so3_feature_dimension(gp_setup["M2"]),
        "Dv": euclidean_feature_dimension(gp_setup["Dv"]),
        "Dw": euclidean_feature_dimension(gp_setup["Dw"]),
        "V": (
            euclidean_feature_dimension(gp_setup["V"]["position"])
            * so3_feature_dimension(gp_setup["V"]["rotation"])
            if potential_uses_rotation
            else euclidean_feature_dimension(gp_setup["V"])
        ),
        "g": (
            euclidean_feature_dimension(gp_setup["g"]["position"])
            * so3_feature_dimension(gp_setup["g"]["rotation"])
        ),
    }


def initialize_variational_parameters(
    key: Array,
    gp_setup: dict[str, Any],
    *,
    initial_log_standard_deviation: float = POSTERIOR_INITIAL_LOG_STD,
    inverse_mass_2_initial_value: float = 1.0,
    dissipation_v_initial_value: float = 1.0,
    dissipation_w_initial_value: float = 1.0,
) -> dict[str, dict[str, Array]]:
    # An SPD operator starts at c*I: the three log-diagonal levels of L are log(sqrt(c)).
    def spd_level(initial_value: float) -> Array:
        return jnp.asarray(
            [0.5 * np.log(initial_value)] * 3 + [0.0] * 3, dtype=jnp.float32
        )

    levels = {
        "M2": spd_level(inverse_mass_2_initial_value),
        "Dv": spd_level(dissipation_v_initial_value),
        "Dw": spd_level(dissipation_w_initial_value),
    }
    if "_exact_specs" in gp_setup:
        specifications = gp_setup.pop("_exact_specs")
        settings = gp_setup.pop("_exact_settings")
        keys = jax.random.split(key, len(GP_NAMES))
        models = {}
        result = {}
        for name, model_key in zip(GP_NAMES, keys):
            input_dim, output_dim, feature_count = specifications[name]
            model = GP_Model(
                model_key,
                input_dim=int(input_dim),
                output_dim=int(output_dim),
                n_matern_features=int(feature_count),
                nu=settings["nu"],
                ell_m=settings["ell_m"],
                period=settings["period"],
                ell_p=settings["ell_p"],
                m_max=settings["m_max"],
                periodic_dim=settings["periodic_dim"],
            )
            models[name] = model
            result[name] = {
                "mean": model.w_mean,
                "log_std": jnp.full(
                    model.w_mean.shape, initial_log_standard_deviation
                ),
            }
            if name == "M1":
                result[name]["log_level"] = jnp.zeros((), dtype=jnp.float32)
            if name in levels:
                result[name]["level"] = levels[name]
            if name == "g":
                result[name]["level"] = jnp.zeros((6, 4), dtype=jnp.float32)
            if name == "V":
                # Linear level of the potential: V(x) = lambda_V^T x + GP(x); only grad V enters the dynamics.
                result[name]["level"] = jnp.zeros((3,), dtype=jnp.float32)
        gp_setup["_exact_models"] = models
        return result

    dimensions = feature_dimensions(gp_setup)
    keys = jax.random.split(key, len(GP_NAMES))
    result: dict[str, dict[str, Array]] = {}
    for name, subnet_key in zip(GP_NAMES, keys):
        shape = (dimensions[name], OUTPUT_DIMENSIONS[name])
        length_scale_count = (
            2
            if name == "g" or (name == "V" and "position" in gp_setup["V"])
            else 1
        )
        result[name] = {
            "mean": jax.random.normal(subnet_key, shape) / float(dimensions[name]),
            "log_std": jnp.full(shape, initial_log_standard_deviation),
            "raw_length_scale": raw_length_scale(
                jnp.ones(length_scale_count, dtype=jnp.float32)
            ),
        }
        if name == "M1":
            result[name]["log_level"] = jnp.zeros((), dtype=jnp.float32)
        if name in levels:
            result[name]["level"] = levels[name]
        if name == "g":
            result[name]["level"] = jnp.zeros((6, 4), dtype=jnp.float32)
    return result


def length_scale_values(
    variational: Mapping[str, Mapping[str, Array]],
) -> dict[str, Array]:
    return {
        name: positive_length_scale(variational[name]["raw_length_scale"])
        for name in GP_NAMES
    }


def sample_weights(
    variational: Mapping[str, Mapping[str, Array]], key: Array | None,
) -> dict[str, Any]:
    if key is None:
        result = {name: variational[name]["mean"] for name in GP_NAMES}
    else:
        keys = jax.random.split(key, len(GP_NAMES))
        result = {
            name: variational[name]["mean"]
            + jnp.exp(variational[name]["log_std"])
            * jax.random.normal(subnet_key, variational[name]["mean"].shape)
            for name, subnet_key in zip(GP_NAMES, keys)
        }
    result["_M1_log_level"] = variational["M1"]["log_level"]
    result["_M2_level"] = variational["M2"]["level"]
    result["_Dv_level"] = variational["Dv"]["level"]
    result["_Dw_level"] = variational["Dw"]["level"]
    result["_g_level"] = variational["g"]["level"]
    # Older checkpoints have no V level; treat it as zero.
    result["_V_level"] = variational["V"].get("level", jnp.zeros((3,), dtype=jnp.float32))
    if "raw_length_scale" in variational["M1"]:
        result["_length_scales"] = length_scale_values(variational)
    return result


def kl_divergence(variational: Mapping[str, Mapping[str, Array]]) -> Array:
    total = jnp.asarray(0.0, dtype=jnp.float32)
    for name in GP_NAMES:
        mean = variational[name]["mean"]
        log_std = variational[name]["log_std"]
        variance = jnp.exp(2.0 * log_std)
        total = total + 0.5 * jnp.sum(
            variance + jnp.square(mean) - 1.0 - 2.0 * log_std
        )
    return total


def features(
    name: str, weights: Mapping[str, Any], gp_setup: Mapping[str, Any], value: Array,
) -> Array:
    if "_exact_models" in gp_setup:
        from .network import exact_gp_input

        model = gp_setup["_exact_models"][name]
        model_input = exact_gp_input(name, value, gp_setup)
        matern_features = model.matern(model_input)
        periodic_features = model.periodic(model_input[model.periodic_dim])
        return jnp.outer(matern_features, periodic_features).reshape(-1)

    scales = weights["_length_scales"][name]
    if name in ("M1", "Dv", "Dw"):
        return euclidean_features(gp_setup[name], value, scales[0])
    if name == "M2":
        return so3_features(gp_setup[name], value, scales[0])
    if name == "V" and "position" not in gp_setup["V"]:
        return euclidean_features(gp_setup["V"], value[:3], scales[0])
    if name in ("V", "g"):
        return se3_features(gp_setup[name], value, scales)
    raise KeyError(name)


def raw_output(
    name: str, weights: Mapping[str, Any], gp_setup: Mapping[str, Any], value: Array,
) -> Array:
    return features(name, weights, gp_setup, value) @ weights[name]


def physical_inverse_mass_targets(gp_setup: Mapping[str, Any]) -> tuple[Array, Array]:
    physics = gp_setup["_physics"]
    return (
        jnp.eye(3, dtype=jnp.float32) / physics["mass"],
        jnp.diag(1.0 / physics["inertia_diagonal"]),
    )
