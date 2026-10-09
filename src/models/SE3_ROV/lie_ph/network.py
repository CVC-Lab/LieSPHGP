"""Port-Hamiltonian SDE on SE(3) x R^6 with variational-GP subnetworks, BlueROV2 version. No physical constant anywhere.

State s = [x (3, world), vec(R) (9, row-major), v (3, body), omega (3, body), u (n_u)];
momenta p_v = M1 v, p_w = M2 omega, H = 1/2 p_v.M1^-1(x) p_v + 1/2 p_w.M2^-1(R) p_w + V(x, R).

    dx = R v dt,   dR = R hat(omega) dt
    dp_v = ( p_v x omega - R^T dV/dx - D_v(v) v + g_F(R, x) u ) dt
    dp_w = ( p_w x omega + p_v x v + sum_i r_i x dH/dr_i - D_w(omega) omega + g_tau(R, x) u ) dt
    d(v, omega) = [M^-1 dp + (dM^-1/dt) p] + Sigma dW      (Sigma diagonal, 6 numbers, twist units m s^-1.5 / rad s^-1.5)

Every subnetwork is a learned level plus a random-feature variational GP residual GP(x) = phi(x) w (features.py,
smooth Matern features) with q(w) = N(m, diag s^2) and prior N(0, I):

    M1^-1(x)     = exp(lambda_1 + GP_M1(x)) I3          : level ()      input x (3)     model.M1_form: scalar
                 = L L^T (exp on the diagonal of L)     : level (6,)    input x (3)     model.M1_form: spd
    M2^-1(R)     = L L^T                                : level (6,)    input vec(R) (9)
    D_v(v)       = L L^T                                : level (6,)    input v (3)
    D_w(omega)   = L L^T                                : level (6,)    input omega (3)
    V(x, R)      = lambda_V . x + GP_V(x)               : level (3,)    input x (3)     model.V_rotation: false
                 = lambda_V . x + lambda_R . vec(R) + GP_V(x)   : level (12,)          model.V_rotation: true
    g(R, x)      = Lambda_g + GP_g(R, x)  (6 x n_u)     : level (6, n_u) input [vec(R), x] (12)

Differences from src/models/SE3_Quadrotor/lie_ph/network.py (with M1_form scalar, V_rotation false and n_u = 4
it is that model exactly):
  - n_u = model.control_dim inputs (the BlueROV2's 8 raw thruster commands) instead of the quadrotor's 4-wrench.
  - M1_form spd: added mass makes the translational inertia anisotropic (the BlueROV2 Heavy's is about
    diag(19.9, 20.6, 32.2) kg), and with M1^-1 = mu I3 the Munk moment p_v x v is identically zero.
  - V_rotation true: the buoyancy acts above the centre of gravity, so V has a term linear in vec(R) (B e3^T R r_b).
    Without it the model has no restoring torque at all. Only the form is given (a zero-initialised linear term); its
    value is learned.

Model variants (config model.family / model.wind; defaults gp / true), integrated with Lie-IMEX:
    Lie-PH-GP-SDE  family gp, wind true            (Lie-PH-GP-ODE, Lie-PH-NN-SDE/ODE: family / wind as the quadrotor)
The PH-NODE baseline (../ph_node) reuses NNSE3Model with RK4.
family nn replaces every level + GP by a tanh MLP (Xavier start, no floor, no prior) on the same inputs:
M1^-1 = exp(MLP(x)) I3 or L L^T from MLP(x), M2^-1 = L L^T from MLP(vec R), D_v / D_w = L L^T from MLP(v) / MLP(omega),
V = MLP(x) or MLP([x, vec R]), g = MLP([vec R, x]) (6 x n_u). wind false removes Sigma.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .features import features, initialize_features

Array = jax.Array


def m1_spd(model_config: Mapping[str, Any]) -> bool:
    form = str(model_config.get("M1_form", "scalar"))
    if form not in ("scalar", "spd"):
        raise ValueError("model.M1_form must be scalar or spd")
    return form == "spd"


def v_rotation(model_config: Mapping[str, Any]) -> bool:
    return bool(model_config.get("V_rotation", False))


def gp_specs(model_config: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    """(input width, output width) of every GP subnetwork."""
    control_dim = int(model_config.get("control_dim", 4))
    return {"M1": (3, 6 if m1_spd(model_config) else 1), "M2": (9, 6), "Dv": (3, 6), "Dw": (3, 6), "V": (3, 1),
            "g": (12, 6 * control_dim)}


def nn_specs(model_config: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    """The MLP family: same widths, except V sees [x, vec R] when it may depend on the rotation."""
    specs = dict(gp_specs(model_config))
    if v_rotation(model_config):
        specs["V"] = (12, 1)
    return specs


SUBNETWORKS = ("M1", "M2", "Dv", "Dw", "V", "g")


def model_family(model_config: Mapping[str, Any]) -> str:
    family = str(model_config.get("family", "gp"))
    if family not in ("gp", "nn"):
        raise ValueError("model.family must be gp or nn")
    return family


def has_wind(model_config: Mapping[str, Any]) -> bool:
    return bool(model_config.get("wind", True))


def initialize_mlp(key: Array, widths: list[int], gain: float | None = None) -> dict[str, Any]:
    """tanh MLP, Xavier-normal weights, zero biases (the usual neural-network start). ``gain`` (default None = 1)
    multiplies every weight matrix: the original PH-NODE code's init_gain (LieGroupHamDL SE3HamNODE.py: 0.001 on every
    layer of every network), which starts the operators near their constant output."""
    layers = []
    for index, (fan_in, fan_out) in enumerate(zip(widths[:-1], widths[1:])):
        std = (2.0 / (fan_in + fan_out)) ** 0.5 * (1.0 if gain is None else float(gain))
        layers.append({"w": std * jax.random.normal(jax.random.fold_in(key, index), (fan_in, fan_out), jnp.float32),
                       "b": jnp.zeros((fan_out,), jnp.float32)})
    return {"layers": layers}


def mlp(net: Mapping[str, Any], value: Array) -> Array:
    hidden = value
    for layer in net["layers"][:-1]:
        hidden = jnp.tanh(hidden @ layer["w"] + layer["b"])
    return hidden @ net["layers"][-1]["w"] + net["layers"][-1]["b"]


def build_gp_setup(model_config: Mapping[str, Any], key: Array) -> dict[str, Any]:
    specs = gp_specs(model_config)
    keys = jax.random.split(key, len(specs))
    setup: dict[str, Any] = {name: initialize_features(subkey, width, int(model_config["feature_count"]),
                                                       float(model_config["matern_smoothness"]),
                                                       float(model_config["matern_length_scale"]))
                             for (name, (width, _)), subkey in zip(specs.items(), keys)}
    # static architecture markers (key presence, never traced values)
    setup["_options"] = {"M1_spd": m1_spd(model_config), "V_rotation": v_rotation(model_config),
                         "control_dim": int(model_config.get("control_dim", 4))}
    return setup


def _spd_level(value: float) -> Array:
    return jnp.asarray([0.5 * np.log(value)] * 3 + [0.0] * 3, dtype=jnp.float32)


def initialize_parameters(model_config: Mapping[str, Any], setup: Mapping[str, Any], key: Array,
                          with_likelihood: bool = True) -> dict[str, Any]:
    """Trainable parameters: per GP {mean, log_std, level}; observation and process noise scales.

    model.initial_randomness = s (optional, default 0 = the fixed neutral start): every level and both noise scales
    start at their neutral value plus s * N(0, 1), drawn from ``key`` (seed-dependent). Same rule as the pendulum."""
    log_std = float(model_config["initial_log_std"])
    spread = float(model_config.get("initial_randomness", 0.0))
    control_dim = int(model_config.get("control_dim", 4))

    def noise(index: int, shape) -> Array:
        return spread * jax.random.normal(jax.random.fold_in(key, 1000 + index), shape, jnp.float32)

    params: dict[str, Any] = {}
    if model_family(model_config) == "nn":
        hidden, depth = int(model_config.get("nn_hidden", 20)), int(model_config.get("nn_layers", 2))
        gain = model_config.get("nn_init_gain")                # None = Xavier; 0.001 = the original PH-NODE start
        for index, (name, (width_in, width_out)) in enumerate(nn_specs(model_config).items()):
            params[name] = initialize_mlp(jax.random.fold_in(key, index), [width_in] + [hidden] * depth + [width_out], gain)
        _noise_parameters(params, model_config, noise, with_likelihood)
        return params

    m_inverse = float(model_config["initial_M_inverse"])
    levels = {
        "M1": (_spd_level(m_inverse) + noise(0, (6,)) if m1_spd(model_config)
               else jnp.asarray(np.log(m_inverse), jnp.float32) + noise(0, ())),
        "M2": _spd_level(m_inverse) + noise(1, (6,)),
        "Dv": _spd_level(float(model_config["initial_D"])) + noise(2, (6,)),
        "Dw": _spd_level(float(model_config["initial_D"])) + noise(3, (6,)),
        "V": noise(4, (12 if v_rotation(model_config) else 3,)),
        "g": noise(5, (6, control_dim)),
    }
    specs = gp_specs(model_config)
    for (name, (_, outputs)), subkey in zip(specs.items(), jax.random.split(key, len(specs))):
        count = int(setup[name]["phases"].shape[0])
        mean = jax.random.normal(subkey, (count, outputs), jnp.float32) / count
        if name == "g":
            mean = mean * float(model_config["initial_control_gp_mean_scale"])
        params[name] = {"mean": mean, "log_std": jnp.full(mean.shape, log_std, jnp.float32), "level": levels[name]}
    _noise_parameters(params, model_config, noise, with_likelihood)
    return params


def _noise_parameters(params, model_config, noise, with_likelihood: bool) -> None:
    """sigma_obs (EKF loss only, [position, attitude, linear velocity, angular velocity]) and Sigma (wind models only)."""
    if with_likelihood:
        obs = float(np.log(float(model_config["initial_observation_sigma"])))
        params["likelihood"] = {"log_sigma": jnp.full((4,), obs, jnp.float32) + noise(6, (4,))}
    if has_wind(model_config):
        params["process"] = {"log_sigma": jnp.full((6,), float(np.log(float(model_config["initial_diffusion"]))),
                                                   jnp.float32) + noise(7, (6,))}


def sample_weights(params: Mapping[str, Any], key: Array | None) -> dict[str, Any]:
    names = SUBNETWORKS
    if key is None:
        weights = {name: params[name]["mean"] for name in names}
    else:
        keys = jax.random.split(key, len(names))
        weights = {name: params[name]["mean"] + jnp.exp(params[name]["log_std"])
                   * jax.random.normal(subkey, params[name]["mean"].shape, params[name]["mean"].dtype)
                   for name, subkey in zip(names, keys)}
    weights["levels"] = {name: params[name]["level"] for name in names}
    if "process" in params:
        weights["process_log_sigma"] = params["process"]["log_sigma"]
    return weights


def kl_divergence(params: Mapping[str, Any]) -> Array:
    total = jnp.zeros((), jnp.float32)
    for name in SUBNETWORKS:
        if "mean" not in params[name]:                     # NN family: no weight posterior, no KL
            continue
        mean, log_std = params[name]["mean"], params[name]["log_std"]
        total = total + 0.5 * jnp.sum(jnp.exp(2.0 * log_std) + mean * mean - 1.0 - 2.0 * log_std)
    return total


def spd_from_levels(value: Array) -> Array:
    diagonal = jnp.exp(value[:3])
    zero = jnp.zeros((), value.dtype)
    lower = jnp.stack([
        jnp.stack([diagonal[0], zero, zero]),
        jnp.stack([value[3], diagonal[1], zero]),
        jnp.stack([value[4], value[5], diagonal[2]]),
    ])
    return lower @ lower.T


class SE3Model:
    """One GP weight sample: the complete vector field, damping and diffusion of the SDE."""

    def __init__(self, weights: Mapping[str, Any], setup: Mapping[str, Any]):
        self.weights = weights
        self.setup = setup
        self.options = setup["_options"]

    def _gp(self, name: str, value: Array) -> Array:
        return features(self.setup[name], value) @ self.weights[name]

    def inverse_mass_1(self, position: Array) -> Array:
        if self.options["M1_spd"]:
            return spd_from_levels(self.weights["levels"]["M1"] + self._gp("M1", position))
        return jnp.exp(self.weights["levels"]["M1"] + self._gp("M1", position)[0]) * jnp.eye(3, dtype=position.dtype)

    def inverse_mass_2(self, rotation_flat: Array) -> Array:
        return spd_from_levels(self.weights["levels"]["M2"] + self._gp("M2", rotation_flat))

    def dissipation_v(self, velocity: Array) -> Array:
        return spd_from_levels(self.weights["levels"]["Dv"] + self._gp("Dv", velocity))

    def dissipation_w(self, omega: Array) -> Array:
        return spd_from_levels(self.weights["levels"]["Dw"] + self._gp("Dw", omega))

    def potential(self, position: Array, rotation_flat: Array) -> Array:
        level = self.weights["levels"]["V"]
        value = level[:3] @ position + self._gp("V", position)[0]
        if self.options["V_rotation"]:
            value = value + level[3:12] @ rotation_flat
        return value

    def control_matrix(self, rotation_flat: Array, position: Array) -> Array:
        value = jnp.concatenate([rotation_flat, position])
        return self.weights["levels"]["g"] + self._gp("g", value).reshape(6, -1)

    def diffusion(self) -> Array:
        """(6,) twist diffusion per body axis: [v_x, v_y, v_z, omega_x, omega_y, omega_z]; zero for the ODE variants."""
        if "process_log_sigma" not in self.weights:
            return jnp.zeros((6,), jnp.float32)
        return jnp.exp(self.weights["process_log_sigma"])

    def hamiltonian(self, pose_momenta: Array) -> Array:
        position, rotation_flat = pose_momenta[:3], pose_momenta[3:12]
        momentum_v, momentum_w = pose_momenta[12:15], pose_momenta[15:18]
        return (0.5 * momentum_v @ self.inverse_mass_1(position) @ momentum_v
                + 0.5 * momentum_w @ self.inverse_mass_2(rotation_flat) @ momentum_w
                + self.potential(position, rotation_flat))

    def vector_field(self, state: Array) -> Array:
        """(18,) [dx, dvec(R), dv, domega] of the drift, damping included."""
        position, rotation_flat = state[:3], state[3:12]
        velocity, omega, control = state[12:15], state[15:18], state[18:]
        inverse_v, inverse_w = self.inverse_mass_1(position), self.inverse_mass_2(rotation_flat)
        momentum_v = jnp.linalg.solve(inverse_v, velocity)
        momentum_w = jnp.linalg.solve(inverse_w, omega)
        gradient = jax.grad(self.hamiltonian)(jnp.concatenate([position, rotation_flat, momentum_v, momentum_w]))
        d_h_dx, d_h_dr = gradient[:3], gradient[3:12].reshape(3, 3)
        d_h_dpv, d_h_dpw = gradient[12:15], gradient[15:18]
        rotation = rotation_flat.reshape(3, 3)
        dx = rotation @ d_h_dpv
        dr = jnp.cross(rotation, d_h_dpw[None, :], axis=-1).reshape(9)
        wrench = self.control_matrix(rotation_flat, position) @ control
        dpv = (jnp.cross(momentum_v, d_h_dpw) - rotation.T @ d_h_dx
               - self.dissipation_v(velocity) @ d_h_dpv + wrench[:3])
        dpw = (jnp.cross(momentum_w, d_h_dpw) + jnp.cross(momentum_v, d_h_dpv)
               + jnp.cross(rotation, d_h_dr, axis=-1).sum(axis=0)
               - self.dissipation_w(omega) @ d_h_dpw + wrench[3:6])
        d_inverse_v = jax.jvp(self.inverse_mass_1, (position,), (dx,))[1]
        d_inverse_w = jax.jvp(self.inverse_mass_2, (rotation_flat,), (dr,))[1]
        dv = inverse_v @ dpv + d_inverse_v @ momentum_v
        dw = inverse_w @ dpw + d_inverse_w @ momentum_w
        return jnp.concatenate([dx, dr, dv, dw])

    def effective_damping(self, state: Array) -> tuple[Array, Array]:
        """(M1^-1 D_v, M2^-1 D_w), the matrices the IMEX step treats implicitly."""
        return (self.inverse_mass_1(state[:3]) @ self.dissipation_v(state[12:15]),
                self.inverse_mass_2(state[3:12]) @ self.dissipation_w(state[15:18]))


class NNSE3Model(SE3Model):
    """Same port-Hamiltonian structure and vector field, with tanh MLP subnetworks instead of level + GP."""

    def inverse_mass_1(self, position: Array) -> Array:
        if self.options["M1_spd"]:
            return spd_from_levels(mlp(self.weights["M1"], position))
        return jnp.exp(mlp(self.weights["M1"], position)[0]) * jnp.eye(3, dtype=position.dtype)

    def inverse_mass_2(self, rotation_flat: Array) -> Array:
        return spd_from_levels(mlp(self.weights["M2"], rotation_flat))

    def dissipation_v(self, velocity: Array) -> Array:
        return spd_from_levels(mlp(self.weights["Dv"], velocity))

    def dissipation_w(self, omega: Array) -> Array:
        return spd_from_levels(mlp(self.weights["Dw"], omega))

    def potential(self, position: Array, rotation_flat: Array) -> Array:
        value = jnp.concatenate([position, rotation_flat]) if self.options["V_rotation"] else position
        return mlp(self.weights["V"], value)[0]

    def control_matrix(self, rotation_flat: Array, position: Array) -> Array:
        return mlp(self.weights["g"], jnp.concatenate([rotation_flat, position])).reshape(6, -1)


def model_from_params(params: Mapping[str, Any], setup: Mapping[str, Any], key: Array | None,
                      model_config: Mapping[str, Any] | None = None) -> SE3Model:
    """The model of any variant; ``key`` samples GP weights (ignored by the NN family)."""
    if "layers" in params["M1"]:
        return NNSE3Model(nn_weights(params), setup)
    return SE3Model(sample_weights(params, key), setup)


def nn_weights(params: Mapping[str, Any]) -> dict[str, Any]:
    """The weights an NNSE3Model reads: the MLP subnetworks (and Sigma for the wind models)."""
    weights = {name: params[name] for name in SUBNETWORKS}
    if "process" in params:
        weights["process_log_sigma"] = params["process"]["log_sigma"]
    return weights
