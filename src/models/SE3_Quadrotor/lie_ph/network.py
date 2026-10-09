"""Port-Hamiltonian SDE on SE(3) x R^6 with variational-GP subnetworks. No physical constant appears anywhere.

State s = [x (3, world), vec(R) (9, row-major), v (3, body), omega (3, body), u (4)];
momenta p_v = M1 v, p_w = M2 omega, H = 1/2 p_v.M1^-1(x) p_v + 1/2 p_w.M2^-1(R) p_w + V(x).

    dx = R v dt,   dR = R hat(omega) dt
    dp_v = ( p_v x omega - R^T dV/dx - D_v(v) v + g_F(R, x) u ) dt
    dp_w = ( p_w x omega + p_v x v + sum_i r_i x dH/dr_i - D_w(omega) omega + g_tau(R, x) u ) dt
    d(v, omega) = [M^-1 dp + (dM^-1/dt) p] + Sigma dW      (Sigma diagonal, 6 numbers, twist units m s^-1.5 / rad s^-1.5)

Every subnetwork is a learned level plus a random-feature variational GP residual GP(x) = phi(x) w (features.py,
smooth Matern features) with q(w) = N(m, diag s^2) and prior N(0, I):

    M1^-1(x)     = exp(lambda_1 + GP_M1(x)) I3          : level ()      input x (3)
    M2^-1(R)     = L L^T (exp on the diagonal of L)     : level (6,)    input vec(R) (9)
    D_v(v)       = L L^T                                : level (6,)    input v (3)
    D_w(omega)   = L L^T                                : level (6,)    input omega (3)
    V(x)         = lambda_V . x + GP_V(x)               : level (3,)    input x (3)
    g(R, x)      = Lambda_g + GP_g(R, x)  (6 x 4)       : level (6, 4)  input [vec(R), x] (12)

Same construction as src/models/3D_SO3_Windy_Pendulum/lie_ph/network.py; the levels start at neutral
values (M^-1 = c_M I, D = c_D I, lambda_V = 0, Lambda_g = 0). Only products such as M^-1 D, M^-1 g, M^-1 grad V and
M^-1 Sigma are identifiable (one scale gauge per block).

Model variants (config model.family / model.wind; defaults gp / true), all integrated with Lie-IMEX:
    Lie-PH-GP-SDE  family gp, wind true            Lie-PH-GP-ODE  family gp, wind false
    Lie-PH-NN-SDE  family nn, wind true            Lie-PH-NN-ODE  family nn, wind false
family nn replaces every level + GP by a tanh MLP (Xavier start, no floor, no prior) on the same inputs:
M1^-1 = exp(MLP(x)) I3, M2^-1 = L L^T from MLP(vec R), D_v / D_w = L L^T from MLP(v) / MLP(omega), V = MLP(x),
g = MLP([vec R, x]) (6 x 4). wind false removes Sigma (no process-noise parameter at all).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .features import features, initialize_features

Array = jax.Array

GP_SPECS = {"M1": (3, 1), "M2": (9, 6), "Dv": (3, 6), "Dw": (3, 6), "V": (3, 1), "g": (12, 24)}
NN_OUTPUTS = GP_SPECS                                   # same (input width, output width) for the MLP family


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


def build_gp_setup(model_config: Mapping[str, Any], key: Array) -> dict[str, dict[str, Array]]:
    keys = jax.random.split(key, len(GP_SPECS))
    return {name: initialize_features(subkey, width, int(model_config["feature_count"]),
                                      float(model_config["matern_smoothness"]),
                                      float(model_config["matern_length_scale"]))
            for (name, (width, _)), subkey in zip(GP_SPECS.items(), keys)}


def _spd_level(value: float) -> Array:
    return jnp.asarray([0.5 * np.log(value)] * 3 + [0.0] * 3, dtype=jnp.float32)


def initialize_parameters(model_config: Mapping[str, Any], setup: Mapping[str, Any], key: Array,
                          with_likelihood: bool = True) -> dict[str, Any]:
    """Trainable parameters: per GP {mean, log_std, level}; observation and process noise scales.

    model.initial_randomness = s (optional, default 0 = the fixed neutral start): every level and both noise scales
    start at their neutral value plus s * N(0, 1), drawn from ``key`` (seed-dependent). Same rule as the pendulum."""
    log_std = float(model_config["initial_log_std"])
    spread = float(model_config.get("initial_randomness", 0.0))

    def noise(index: int, shape) -> Array:
        return spread * jax.random.normal(jax.random.fold_in(key, 1000 + index), shape, jnp.float32)

    params: dict[str, Any] = {}
    if model_family(model_config) == "nn":
        hidden, depth = int(model_config.get("nn_hidden", 20)), int(model_config.get("nn_layers", 2))
        for index, (name, (width_in, width_out)) in enumerate(NN_OUTPUTS.items()):
            params[name] = initialize_mlp(jax.random.fold_in(key, index), [width_in] + [hidden] * depth + [width_out])
        _noise_parameters(params, model_config, noise, with_likelihood)
        return params

    levels = {
        "M1": jnp.asarray(np.log(float(model_config["initial_M_inverse"])), jnp.float32) + noise(0, ()),
        "M2": _spd_level(float(model_config["initial_M_inverse"])) + noise(1, (6,)),
        "Dv": _spd_level(float(model_config["initial_D"])) + noise(2, (6,)),
        "Dw": _spd_level(float(model_config["initial_D"])) + noise(3, (6,)),
        "V": noise(4, (3,)),
        "g": noise(5, (6, 4)),
    }
    for (name, (_, outputs)), subkey in zip(GP_SPECS.items(), jax.random.split(key, len(GP_SPECS))):
        count = int(setup[name]["phases"].shape[0])
        mean = jax.random.normal(subkey, (count, outputs), jnp.float32) / count
        if name == "g":
            mean = mean * float(model_config["initial_control_gp_mean_scale"])
        params[name] = {"mean": mean, "log_std": jnp.full(mean.shape, log_std, jnp.float32), "level": levels[name]}
        # model.learn_gp_hyperparameters (optional, default false): every GP residual gets a trainable amplitude tau
        # and length-scale factor ell, residual = tau * phi(x / ell)^T w, learned through the ELBO (type-II maximum
        # likelihood, no hyper-prior) so an unneeded residual can be switched off (tau -> 0) at zero KL. Same as the
        # pendulum package (5 Oct 2026). model.fixed_hyperparameter_gps (optional): GPs that keep tau = ell = 1.
        if bool(model_config.get("learn_gp_hyperparameters", False)) and name not in tuple(model_config.get("fixed_hyperparameter_gps", ())):
            params[name]["log_amplitude"] = jnp.zeros((), jnp.float32)       # tau = 1
            params[name]["log_length"] = jnp.zeros((), jnp.float32)          # ell = 1 (x matern_length_scale)
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
    total = jnp.zeros((), jnp.float32)
    for name in GP_SPECS:
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

    def __init__(self, weights: Mapping[str, Any], setup: Mapping[str, Any], eigenvalue_floor: float = 0.0):
        self.weights = weights
        self.setup = setup
        # model.spd_eigenvalue_floor (optional, default 0 = off): M2^-1 = L L^T + floor I, so the rotational inverse
        # mass cannot become singular (the September IDSIA package's mass_epsilon, 0.01 there). A conditioning floor,
        # not a fitted constant.
        self.eigenvalue_floor = float(eigenvalue_floor)

    def _floored(self, matrix: Array) -> Array:
        if self.eigenvalue_floor == 0.0:
            return matrix
        return matrix + self.eigenvalue_floor * jnp.eye(3, dtype=matrix.dtype)

    def _gp(self, name: str, value: Array) -> Array:
        hyper = self.weights.get("hyper", {}).get(name)
        if hyper is None:                                    # fixed hyperparameters (tau = ell = 1)
            return features(self.setup[name], value) @ self.weights[name]
        log_amplitude, log_length = hyper
        return jnp.exp(log_amplitude) * (features(self.setup[name], value * jnp.exp(-log_length)) @ self.weights[name])

    def inverse_mass_1(self, position: Array) -> Array:
        return jnp.exp(self.weights["levels"]["M1"] + self._gp("M1", position)[0]) * jnp.eye(3, dtype=position.dtype)

    def inverse_mass_2(self, rotation_flat: Array) -> Array:
        return self._floored(spd_from_levels(self.weights["levels"]["M2"] + self._gp("M2", rotation_flat)))

    def dissipation_v(self, velocity: Array) -> Array:
        return spd_from_levels(self.weights["levels"]["Dv"] + self._gp("Dv", velocity))

    def dissipation_w(self, omega: Array) -> Array:
        return spd_from_levels(self.weights["levels"]["Dw"] + self._gp("Dw", omega))

    def potential(self, position: Array) -> Array:
        return self.weights["levels"]["V"] @ position + self._gp("V", position)[0]

    def control_matrix(self, rotation_flat: Array, position: Array) -> Array:
        value = jnp.concatenate([rotation_flat, position])
        return self.weights["levels"]["g"] + self._gp("g", value).reshape(6, 4)

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
                + self.potential(position))

    def vector_field(self, state: Array) -> Array:
        """(18,) [dx, dvec(R), dv, domega] of the drift, damping included."""
        position, rotation_flat = state[:3], state[3:12]
        velocity, omega, control = state[12:15], state[15:18], state[18:22]
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
        return jnp.exp(mlp(self.weights["M1"], position)[0]) * jnp.eye(3, dtype=position.dtype)

    def inverse_mass_2(self, rotation_flat: Array) -> Array:
        return self._floored(spd_from_levels(mlp(self.weights["M2"], rotation_flat)))

    def dissipation_v(self, velocity: Array) -> Array:
        return spd_from_levels(mlp(self.weights["Dv"], velocity))

    def dissipation_w(self, omega: Array) -> Array:
        return spd_from_levels(mlp(self.weights["Dw"], omega))

    def potential(self, position: Array) -> Array:
        return mlp(self.weights["V"], position)[0]

    def control_matrix(self, rotation_flat: Array, position: Array) -> Array:
        return mlp(self.weights["g"], jnp.concatenate([rotation_flat, position])).reshape(6, 4)


def model_from_params(params: Mapping[str, Any], setup: Mapping[str, Any], key: Array | None,
                      model_config: Mapping[str, Any] | None = None) -> SE3Model:
    """The model of any variant; ``key`` samples GP weights (ignored by the NN family)."""
    floor = float((model_config or {}).get("spd_eigenvalue_floor", 0.0))
    if "layers" in params["M1"]:
        return NNSE3Model(nn_weights(params), setup, floor)
    return SE3Model(sample_weights(params, key), setup, floor)


def nn_weights(params: Mapping[str, Any]) -> dict[str, Any]:
    """The weights an NNSE3Model reads: the six MLPs (and Sigma for the wind models)."""
    weights = {name: params[name] for name in NN_OUTPUTS}
    if "process" in params:
        weights["process_log_sigma"] = params["process"]["log_sigma"]
    return weights
