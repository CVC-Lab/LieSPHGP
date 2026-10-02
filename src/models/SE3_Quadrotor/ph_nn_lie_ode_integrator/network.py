"""Readable neural port-Hamiltonian model on SE(3)."""

from __future__ import annotations

from collections.abc import Iterator

import equinox as eqx
import jax
import jax.numpy as jnp


Array = jax.Array


def _init_linear(
    key: Array, input_dim: int, output_dim: int, gain: float
) -> tuple[Array, Array]:
    limit = gain * jnp.sqrt(6.0 / float(input_dim + output_dim))
    weight = jax.random.uniform(
        key, (output_dim, input_dim), minval=-limit, maxval=limit
    )
    return weight, jnp.zeros((output_dim,), dtype=jnp.float32)


class MLP(eqx.Module):
    """Small tanh MLP used by one physical subnetwork."""

    weights: tuple[Array, ...]
    biases: tuple[Array, ...]

    def __init__(
        self, keys: Iterator[Array], dimensions: tuple[int, ...], gain: float
    ):
        layers = [
            _init_linear(next(keys), input_dim, output_dim, gain)
            for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:])
        ]
        self.weights = tuple(weight for weight, _ in layers)
        self.biases = tuple(bias for _, bias in layers)

    def __call__(self, value: Array) -> Array:
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            value = jnp.tanh(weight @ value + bias)
        return self.weights[-1] @ value + self.biases[-1]


class PSDNetwork(eqx.Module):
    """MLP followed by a lower-triangular positive-semidefinite factor."""

    mlp: MLP
    diagonal_shift: Array
    eigenvalue_floor: Array

    def __init__(self, mlp: MLP, diagonal_shift: Array, eigenvalue_floor: Array):
        self.mlp = mlp
        self.diagonal_shift = diagonal_shift
        self.eigenvalue_floor = eigenvalue_floor

    def __call__(self, value: Array) -> Array:
        output = self.mlp(value)
        diagonal = output[:3] + self.diagonal_shift
        lower = jnp.array(
            [
                [diagonal[0], 0.0, 0.0],
                [output[3], diagonal[1], 0.0],
                [output[4], output[5], diagonal[2]],
            ],
            dtype=value.dtype,
        )
        return (
            lower @ lower.T
            + self.eigenvalue_floor * jnp.eye(3, dtype=value.dtype)
        )


class DissipativeSE3HamNODE(eqx.Module):
    """Six-subnetwork port-Hamiltonian quadrotor model.

    State: [x(3), vec(R)(9), v(3), omega(3), u(4)].
    """

    M1_net: PSDNetwork
    M2_net: PSDNetwork
    Dv_net: PSDNetwork
    Dw_net: PSDNetwork
    V_net: MLP
    g_net: MLP
    dissipation_enabled: Array

    def __init__(
        self,
        key: Array,
        *,
        hidden_dim: int,
        initialization_gain: float,
        mass_epsilon: float,
        mass_factor_epsilon: float,
        dissipation_enabled: bool,
    ):
        dimensions = (
            (3, hidden_dim, hidden_dim, hidden_dim, 6),
            (9, hidden_dim, hidden_dim, hidden_dim, 6),
            (3, hidden_dim, hidden_dim, hidden_dim, 6),
            (9, hidden_dim, hidden_dim, hidden_dim, 6),
            (12, hidden_dim, hidden_dim, 1),
            (12, hidden_dim, hidden_dim, 24),
        )
        layer_count = sum(len(shape) - 1 for shape in dimensions)
        keys = iter(jax.random.split(key, layer_count))
        mass_floor = jnp.asarray(mass_epsilon, dtype=jnp.float32)
        mass_shift = jnp.sqrt(
            jnp.asarray(mass_factor_epsilon, dtype=jnp.float32)
        )
        zero = jnp.asarray(0.0, dtype=jnp.float32)

        self.M1_net = PSDNetwork(
            MLP(keys, dimensions[0], initialization_gain), mass_shift, mass_floor
        )
        self.M2_net = PSDNetwork(
            MLP(keys, dimensions[1], initialization_gain), mass_shift, mass_floor
        )
        self.Dv_net = PSDNetwork(
            MLP(keys, dimensions[2], initialization_gain), zero, zero
        )
        self.Dw_net = PSDNetwork(
            MLP(keys, dimensions[3], initialization_gain), zero, zero
        )
        self.V_net = MLP(keys, dimensions[4], initialization_gain)
        self.g_net = MLP(keys, dimensions[5], initialization_gain)
        self.dissipation_enabled = jnp.asarray(
            float(dissipation_enabled), dtype=jnp.float32
        )

    def inverse_mass_1(self, position: Array) -> Array:
        return self.M1_net(position)

    def inverse_mass_2(self, rotation_flat: Array) -> Array:
        return self.M2_net(rotation_flat)

    def dissipation_v(self, position: Array) -> Array:
        return self.dissipation_enabled * self.Dv_net(position)

    def dissipation_w(self, rotation_flat: Array) -> Array:
        return self.dissipation_enabled * self.Dw_net(rotation_flat)

    def potential(self, pose: Array) -> Array:
        return self.V_net(pose).squeeze()

    def control_matrix(self, pose: Array) -> Array:
        return self.g_net(pose).reshape(6, 4)

    def hamiltonian(self, pose_momenta: Array) -> Array:
        position = pose_momenta[:3]
        rotation_flat = pose_momenta[3:12]
        momentum_v = pose_momenta[12:15]
        momentum_w = pose_momenta[15:18]
        return (
            0.5 * momentum_v @ self.inverse_mass_1(position) @ momentum_v
            + 0.5
            * momentum_w
            @ self.inverse_mass_2(rotation_flat)
            @ momentum_w
            + self.potential(pose_momenta[:12])
        )

    def vector_field(self, state: Array) -> Array:
        pose = state[:12]
        position = pose[:3]
        rotation_flat = pose[3:12]
        velocity = state[12:15]
        omega = state[15:18]
        control = state[18:22]

        mass_inv_v = self.inverse_mass_1(position)
        mass_inv_w = self.inverse_mass_2(rotation_flat)
        momentum_v = jnp.linalg.solve(mass_inv_v, velocity)
        momentum_w = jnp.linalg.solve(mass_inv_w, omega)
        pose_momenta = jnp.concatenate([pose, momentum_v, momentum_w])
        gradient = jax.grad(self.hamiltonian)(pose_momenta)
        d_h_dx, d_h_dr = gradient[:3], gradient[3:12]
        d_h_dpv, d_h_dpw = gradient[12:15], gradient[15:18]

        rotation = rotation_flat.reshape(3, 3)
        dx = rotation @ d_h_dpv
        dr = jnp.cross(rotation, d_h_dpw[None, :], axis=-1).reshape(9)
        force_torque = self.control_matrix(pose) @ control
        dpv = (
            jnp.cross(momentum_v, d_h_dpw)
            - rotation.T @ d_h_dx
            - self.dissipation_v(position) @ d_h_dpv
            + force_torque[:3]
        )
        dpw = (
            jnp.cross(momentum_w, d_h_dpw)
            + jnp.cross(momentum_v, d_h_dpv)
            + jnp.cross(rotation, d_h_dr.reshape(3, 3), axis=-1).sum(axis=0)
            - self.dissipation_w(rotation_flat) @ d_h_dpw
            + force_torque[3:6]
        )

        dmass_inv_v = jax.jvp(
            self.inverse_mass_1, (position,), (dx,)
        )[1]
        dmass_inv_w = jax.jvp(
            self.inverse_mass_2, (rotation_flat,), (dr,)
        )[1]
        dv = mass_inv_v @ dpv + dmass_inv_v @ momentum_v
        dw = mass_inv_w @ dpw + dmass_inv_w @ momentum_w
        return jnp.concatenate(
            [dx, dr, dv, dw, jnp.zeros(4, dtype=state.dtype)]
        )

    def effective_damping(self, state: Array) -> tuple[Array, Array]:
        position = state[:3]
        rotation_flat = state[3:12]
        return (
            self.inverse_mass_1(position) @ self.dissipation_v(position),
            self.inverse_mass_2(rotation_flat)
            @ self.dissipation_w(rotation_flat),
        )


# Thin wrappers preserve the existing public function names.
def initialize_parameters(key: Array, **settings) -> DissipativeSE3HamNODE:
    return DissipativeSE3HamNODE(key, **settings)


def load_legacy_parameters(
    model: DissipativeSE3HamNODE, flat: dict[str, Array]
) -> DissipativeSE3HamNODE:
    """Load the former flat-dictionary checkpoint into the readable model."""
    specifications = (
        ("M1_net", "M_net1", 4, True),
        ("M2_net", "M_net2", 4, True),
        ("Dv_net", "Dv_net", 4, True),
        ("Dw_net", "Dw_net", 4, True),
        ("V_net", "V_net", 3, False),
        ("g_net", "g_net.mlp", 3, False),
    )
    for attribute, prefix, layers, is_psd in specifications:
        weights = tuple(flat[f"{prefix}.linear{i}.weight"] for i in range(1, layers + 1))
        biases = tuple(flat[f"{prefix}.linear{i}.bias"] for i in range(1, layers + 1))
        if is_psd:
            model = eqx.tree_at(lambda tree: getattr(tree, attribute).mlp.weights, model, weights)
            model = eqx.tree_at(lambda tree: getattr(tree, attribute).mlp.biases, model, biases)
        else:
            model = eqx.tree_at(lambda tree: getattr(tree, attribute).weights, model, weights)
            model = eqx.tree_at(lambda tree: getattr(tree, attribute).biases, model, biases)
    return model


def inverse_mass_1(model: DissipativeSE3HamNODE, position: Array) -> Array:
    return model.inverse_mass_1(position)


def inverse_mass_2(model: DissipativeSE3HamNODE, rotation_flat: Array) -> Array:
    return model.inverse_mass_2(rotation_flat)


def vector_field_single(model: DissipativeSE3HamNODE, state: Array) -> Array:
    return model.vector_field(state)


def effective_damping_single(
    model: DissipativeSE3HamNODE, state: Array
) -> tuple[Array, Array]:
    return model.effective_damping(state)


vector_field = jax.vmap(vector_field_single, in_axes=(None, 0))
effective_damping_blocks = jax.vmap(
    effective_damping_single, in_axes=(None, 0)
)
