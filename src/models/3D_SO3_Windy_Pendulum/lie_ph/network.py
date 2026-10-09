"""Port-Hamiltonian SDE on SO(3) x R^3 with variational-GP subnetworks. No physical constant appears anywhere.

State s = [vec(R) (9, row-major), omega (3, body), u (3)], momentum p = M(R) omega.

    dR      = R hat(omega) dt
    d omega = [ M^-1 ( p x omega + sum_i r_i x dV/dr_i - D(omega) omega + g(R) u ) + (dM^-1/dt) p ] dt + Sigma dW

(r_i = rows of R). Every subnetwork is a learned level plus a random-feature variational GP residual
GP(x) = phi(x) w (features.py: smooth Matern features on vec(R) or R^n) whose weights w ~ q(w) = N(m, diag s^2)
have the prior N(0, I):

    M^-1(R)   = L L^T, L lower triangular from 6 numbers, exp() on the diagonal : level (6,) + GP_M(R)
    D(omega)  = L L^T (same construction)                                      : level (6,) + GP_D(omega)
    V(R)      = lambda_V . vec(R) + GP_V(R)                                     : level (9,)
    g(R)      = Lambda_g + GP_g(R)  (3 x 3)                                     : level (3, 3)
    Sigma     = diag(exp(log_sigma)) (3,), constant, in twist units (rad s^-1.5)

The levels start at neutral values (M^-1 = c_M I, D = c_D I, lambda_V = 0, Lambda_g = 0) from the config.
(M^-1, V, D, g) -> (c M^-1, V / c, D / c, g / c) leaves the dynamics unchanged, so only the products
M^-1 D, M^-1 g, M^-1 grad V are identifiable; the absolute M^-1 is a gauge choice.

Mirrors src/models/SE3_Quadrotor/lie_ph/network.py (SE(3) x R^6).

Model variants (config model.family / model.wind; defaults gp / true), all integrated with Lie-IMEX:
    Lie-PH-GP-SDE  family gp, wind true            Lie-PH-GP-ODE  family gp, wind false
    Lie-PH-NN-SDE  family nn, wind true            Lie-PH-NN-ODE  family nn, wind false
model.learn_gp_hyperparameters (optional, default false): every GP residual gets a trainable amplitude tau and
length-scale factor ell, residual(x) = tau * phi(x / ell) w with the prior w ~ N(0, I) unchanged (equivalently the prior
w ~ N(0, tau^2 I) on the base kernel with length-scale ell * matern_length_scale). Both are learned through the
ELBO (type-II maximum likelihood, no hyper-prior): a residual the data does not need can be switched off at zero KL
cost (tau -> 0 with the posterior back at the prior), a needed one keeps its amplitude. Start tau = ell = 1 (the
fixed-hyperparameter model).
model.dissipation_input (optional, default "omega"): "state" feeds the dissipation subnetwork [vec(R), omega] (12)
instead of omega (3), i.e. D(q, omega), the same function class as D(q, p) since p = M(q) omega is one-to-one for a
given q (omega is used because the learned momentum is gauge-dependent). GP and NN families.
family nn replaces every level + GP by a tanh MLP (standard Xavier start, no floor, no prior):
M^-1 = L L^T from MLP_M(vec R), D = L L^T from MLP_D(omega), V = MLP_V(vec R), g = MLP_g(vec R) (3 x 3).
wind false removes Sigma (no process noise parameter at all).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .features import features, initialize_features

Array = jax.Array

# name -> (GP input width, output width). Inputs: vec(R) (9) for M, V, g; omega (3) for D.
GP_SPECS = {"M": (9, 6), "D": (3, 6), "V": (9, 1), "g": (9, 9)}
LEVEL_SHAPES = {"M": (6,), "D": (6,), "V": (9,), "g": (3, 3)}


def input_width(name: str, model_config: Mapping[str, Any]) -> int:
    """Input width of a subnetwork: GP_SPECS, except D with model.dissipation_input = state ([vec(R), omega], 12)."""
    choice = str(model_config.get("dissipation_input", "omega"))
    if choice not in ("omega", "state"):
        raise ValueError("model.dissipation_input must be omega or state")
    return 12 if name == "D" and choice == "state" else GP_SPECS[name][0]


def build_gp_setup(model_config: Mapping[str, Any], key: Array) -> dict[str, dict[str, Array]]:
    """The fixed random features of every GP (not trained). Deterministic in ``key``."""
    keys = jax.random.split(key, len(GP_SPECS))
    return {name: initialize_features(subkey, input_width(name, model_config), int(model_config["feature_count"]),
                                      float(model_config["matern_smoothness"]),
                                      float(model_config["matern_length_scale"]))
            for name, subkey in zip(GP_SPECS, keys)}


def _spd_level(value: float) -> Array:
    """Six numbers whose L L^T is value * I."""
    return jnp.asarray([0.5 * np.log(value)] * 3 + [0.0] * 3, dtype=jnp.float32)


NN_OUTPUTS = {"M": (9, 6), "D": (3, 6), "V": (9, 1), "g": (9, 9)}       # (input width, output width)


def model_family(model_config: Mapping[str, Any]) -> str:
    family = str(model_config.get("family", "gp"))
    if family not in ("gp", "nn"):
        raise ValueError("model.family must be gp or nn")
    return family


def has_wind(model_config: Mapping[str, Any]) -> bool:
    return bool(model_config.get("wind", True))


def initialize_mlp(key: Array, widths: list[int]) -> dict[str, Any]:
    """tanh MLP, Xavier-normal weights, zero biases (the usual neural-network start)."""
    layers = []
    for index, (fan_in, fan_out) in enumerate(zip(widths[:-1], widths[1:])):
        std = (2.0 / (fan_in + fan_out)) ** 0.5
        layers.append({"w": std * jax.random.normal(jax.random.fold_in(key, index), (fan_in, fan_out), jnp.float32),
                       "b": jnp.zeros((fan_out,), jnp.float32)})
    return {"layers": layers}


def mlp(net: Mapping[str, Any], value: Array) -> Array:
    hidden = value
    for layer in net["layers"][:-1]:
        hidden = jnp.tanh(hidden @ layer["w"] + layer["b"])
    return hidden @ net["layers"][-1]["w"] + net["layers"][-1]["b"]


def initialize_parameters(model_config: Mapping[str, Any], setup: Mapping[str, Any], key: Array,
                          with_likelihood: bool = True) -> dict[str, Any]:
    """Trainable parameters: per GP {mean, log_std, level}; observation and process noise scales.

    model.initial_randomness = s (optional, default 0 = the fixed neutral start): every level and both noise scales
    start at their neutral value plus s * N(0, 1), drawn from ``key`` (so the start depends on the seed). The levels of
    M^-1 and D are log-Cholesky numbers, so their start scale is exp(2 s eps): within about +-20 % of the neutral
    value for s = 0.1, and 0.14x to 7x (one standard deviation) for s = 1."""
    log_std = float(model_config["initial_log_std"])
    spread = float(model_config.get("initial_randomness", 0.0))

    def noise(index: int, shape) -> Array:
        return spread * jax.random.normal(jax.random.fold_in(key, 1000 + index), shape, jnp.float32)

    params: dict[str, Any] = {}
    if model_family(model_config) == "nn":
        hidden, depth = int(model_config.get("nn_hidden", 20)), int(model_config.get("nn_layers", 2))
        for index, (name, (_, width_out)) in enumerate(NN_OUTPUTS.items()):
            params[name] = initialize_mlp(jax.random.fold_in(key, index),
                                          [input_width(name, model_config)] + [hidden] * depth + [width_out])
        _noise_parameters(params, model_config, noise, with_likelihood)
        return params

    levels = {
        "M": _spd_level(float(model_config["initial_M_inverse"])) + noise(0, LEVEL_SHAPES["M"]),
        "D": _spd_level(float(model_config["initial_D"])) + noise(1, LEVEL_SHAPES["D"]),
        "V": noise(2, LEVEL_SHAPES["V"]),
        "g": noise(3, LEVEL_SHAPES["g"]),
    }
    for (name, (_, outputs)), subkey in zip(GP_SPECS.items(), jax.random.split(key, len(GP_SPECS))):
        count = int(setup[name]["phases"].shape[0])
        mean = jax.random.normal(subkey, (count, outputs), jnp.float32) / count      # small random start
        if name == "g":
            mean = mean * float(model_config["initial_control_gp_mean_scale"])
        params[name] = {"mean": mean, "log_std": jnp.full(mean.shape, log_std, jnp.float32), "level": levels[name]}
        # model.fixed_hyperparameter_gps (optional, e.g. ["D"]): GPs that keep tau = ell = 1 although the others learn theirs
        if bool(model_config.get("learn_gp_hyperparameters", False)) and name not in tuple(model_config.get("fixed_hyperparameter_gps", ())):
            params[name]["log_amplitude"] = jnp.zeros((), jnp.float32)       # tau = 1
            params[name]["log_length"] = jnp.zeros((), jnp.float32)          # ell = 1 (x matern_length_scale)
    _noise_parameters(params, model_config, noise, with_likelihood)
    return params


def _noise_parameters(params, model_config, noise, with_likelihood: bool) -> None:
    """sigma_obs (EKF loss only) and Sigma (wind models only)."""
    if with_likelihood:
        obs = float(np.log(float(model_config["initial_observation_sigma"])))
        params["likelihood"] = {"log_sigma": jnp.full((2,), obs, jnp.float32) + noise(4, (2,))}   # [attitude, angular velocity]
    if has_wind(model_config):
        params["process"] = {"log_sigma": jnp.full((3,), float(np.log(float(model_config["initial_diffusion"]))),
                                                   jnp.float32) + noise(5, (3,))}


def sample_weights(params: Mapping[str, Any], key: Array | None) -> dict[str, Any]:
    """One coherent weight sample of every GP (key None = posterior mean), plus the levels and noise scales."""
    names = tuple(GP_SPECS)
    if key is None:
        weights = {name: params[name]["mean"] for name in names}
    else:
        keys = jax.random.split(key, len(names))
        weights = {name: params[name]["mean"] + jnp.exp(params[name]["log_std"])
                   * jax.random.normal(subkey, params[name]["mean"].shape, params[name]["mean"].dtype)
                   for name, subkey in zip(names, keys)}
    weights["levels"] = {name: params[name]["level"] for name in names}
    weights["hyper"] = {name: (params[name]["log_amplitude"], params[name]["log_length"])
                        for name in names if "log_amplitude" in params[name]}
    if "process" in params:
        weights["process_log_sigma"] = params["process"]["log_sigma"]
    return weights


def kl_divergence(params: Mapping[str, Any]) -> Array:
    """KL(q(w) || N(0, I)) summed over every GP weight."""
    total = jnp.zeros((), jnp.float32)
    for name in GP_SPECS:
        if "mean" not in params[name]:          # neural-network family: no weight posterior, no KL
            continue
        mean, log_std = params[name]["mean"], params[name]["log_std"]
        total = total + 0.5 * jnp.sum(jnp.exp(2.0 * log_std) + mean * mean - 1.0 - 2.0 * log_std)
    return total


def spd_from_levels(value: Array) -> Array:
    """L L^T from six numbers: exp() of the first three on the diagonal, the last three below it."""
    diagonal = jnp.exp(value[:3])
    zero = jnp.zeros((), value.dtype)
    lower = jnp.stack([
        jnp.stack([diagonal[0], zero, zero]),
        jnp.stack([value[3], diagonal[1], zero]),
        jnp.stack([value[4], value[5], diagonal[2]]),
    ])
    return lower @ lower.T


class SO3Model:
    """One GP weight sample: the complete vector field, damping and diffusion of the SDE."""

    def __init__(self, weights: Mapping[str, Any], setup: Mapping[str, Any]):
        self.weights = weights
        self.setup = setup

    def _gp(self, name: str, value: Array) -> Array:
        hyper = self.weights.get("hyper", {}).get(name)
        if hyper is None:                                    # fixed hyperparameters (tau = ell = 1)
            return features(self.setup[name], value) @ self.weights[name]
        log_amplitude, log_length = hyper
        return jnp.exp(log_amplitude) * (features(self.setup[name], value * jnp.exp(-log_length)) @ self.weights[name])

    def inverse_mass(self, rotation_flat: Array) -> Array:
        return spd_from_levels(self.weights["levels"]["M"] + self._gp("M", rotation_flat))

    def dissipation(self, omega: Array, rotation_flat: Array | None = None) -> Array:
        """D(omega), or D(vec(R), omega) when the D features were built for the 12-dim state input."""
        if self.setup["D"]["frequencies"].shape[-1] == 12:
            return spd_from_levels(self.weights["levels"]["D"] + self._gp("D", jnp.concatenate([rotation_flat, omega])))
        return spd_from_levels(self.weights["levels"]["D"] + self._gp("D", omega))

    def potential(self, rotation_flat: Array) -> Array:
        return self.weights["levels"]["V"] @ rotation_flat + self._gp("V", rotation_flat)[0]

    def control_matrix(self, rotation_flat: Array) -> Array:
        return self.weights["levels"]["g"] + self._gp("g", rotation_flat).reshape(3, 3)

    def diffusion(self) -> Array:
        """(3,) twist diffusion per body axis; zero for the wind-free (ODE) variants."""
        if "process_log_sigma" not in self.weights:
            return jnp.zeros((3,), jnp.float32)
        return jnp.exp(self.weights["process_log_sigma"])

    def vector_field(self, state: Array) -> tuple[Array, Array]:
        """(dvec(R)/dt (9,), domega/dt (3,)) of the drift, damping included."""
        rotation_flat, omega, control = state[:9], state[9:12], state[12:15]
        inverse_mass = self.inverse_mass(rotation_flat)
        momentum = jnp.linalg.solve(inverse_mass, omega)

        def hamiltonian(r):
            return 0.5 * momentum @ self.inverse_mass(r) @ momentum + self.potential(r)

        d_h_dr = jax.grad(hamiltonian)(rotation_flat).reshape(3, 3)
        rotation = rotation_flat.reshape(3, 3)
        d_rotation = jnp.cross(rotation, omega[None, :], axis=-1).reshape(9)        # rows of R hat(omega)
        d_momentum = (jnp.cross(momentum, omega)
                      + jnp.cross(rotation, d_h_dr, axis=-1).sum(axis=0)
                      - self.dissipation(omega, rotation_flat) @ omega
                      + self.control_matrix(rotation_flat) @ control)
        d_inverse_mass = jax.jvp(self.inverse_mass, (rotation_flat,), (d_rotation,))[1]
        return d_rotation, inverse_mass @ d_momentum + d_inverse_mass @ momentum

    def effective_damping(self, state: Array) -> Array:
        """M^-1 D, the matrix the IMEX step treats implicitly (domega = -M^-1 D omega from damping)."""
        return self.inverse_mass(state[:9]) @ self.dissipation(state[9:12], state[:9])


class NNSO3Model(SO3Model):
    """Same port-Hamiltonian structure and vector field, with tanh MLP subnetworks instead of level + GP."""

    def inverse_mass(self, rotation_flat: Array) -> Array:
        return spd_from_levels(mlp(self.weights["M"], rotation_flat))

    def dissipation(self, omega: Array, rotation_flat: Array | None = None) -> Array:
        if self.weights["D"]["layers"][0]["w"].shape[0] == 12:
            return spd_from_levels(mlp(self.weights["D"], jnp.concatenate([rotation_flat, omega])))
        return spd_from_levels(mlp(self.weights["D"], omega))

    def potential(self, rotation_flat: Array) -> Array:
        return mlp(self.weights["V"], rotation_flat)[0]

    def control_matrix(self, rotation_flat: Array) -> Array:
        return mlp(self.weights["g"], rotation_flat).reshape(3, 3)


def model_from_params(params: Mapping[str, Any], setup: Mapping[str, Any], key: Array | None,
                      model_config: Mapping[str, Any] | None = None) -> SO3Model:
    """The model of any variant; ``key`` samples GP weights (ignored by the NN family)."""
    if "layers" in params["M"]:
        return NNSO3Model(nn_weights(params), setup)
    return SO3Model(sample_weights(params, key), setup)


def nn_weights(params: Mapping[str, Any]) -> dict[str, Any]:
    """The weights an NNSO3Model reads: the four MLPs (and Sigma for the wind models)."""
    weights = {name: params[name] for name in NN_OUTPUTS}
    if "process" in params:
        weights["process_log_sigma"] = params["process"]["log_sigma"]
    return weights
