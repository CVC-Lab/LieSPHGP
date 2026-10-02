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


# input width of the diffusion subnetwork for each sde.diffusion_input
DIFFUSION_INPUTS = {"constant": 0, "twist_control": 10, "orientation": 9, "full": 22}


def diffusion_features(state: Array, mode: str) -> Array:
    """Input of sigma_net. (a) twist_control: body twist + rotor inputs; (b) orientation: vec(R);
    (c) full: position, vec(R), twist and rotor inputs."""
    if mode == "twist_control":
        return jnp.concatenate([state[12:18], state[18:22]])
    if mode == "orientation":
        return state[3:12]
    if mode == "full":
        return state[:22]
    raise ValueError(f"unknown diffusion input {mode!r}")


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
    # SDE variant: log of the two diffusion scales on the body twist (linear, angular) and of the four
    # observation scales. Trained like every other leaf; they are ordinary fields so checkpoints, the
    # optimiser and jax.tree_util need no special cases.
    process_log_sigma: Array
    log_sigma_observation: Array
    # State-dependent diffusion (config sde.diffusion_input): "constant" keeps the two scales above, added to
    # the body twist (the original model). Any other value adds sigma_net, an MLP whose 6 softplus outputs are
    # the scales of a random body force (3) and torque (3) entering the MOMENTUM equations, so the twist noise
    # is M^-1 Sigma(x) dW (pendulum-style: noise is an unknown wrench through the port). Class-level defaults
    # let checkpoints pickled before this field existed load as the constant model.
    sigma_net: MLP | None = None
    diffusion_input: str = eqx.field(static=True, default="constant")

    def __init__(
        self,
        key: Array,
        *,
        hidden_dim: int,
        initialization_gain: float,
        mass_epsilon: float,
        mass_factor_epsilon: float,
        dissipation_enabled: bool,
        initial_process_sigma: tuple[float, float] = (0.05, 0.05),
        initial_observation_sigma: float = 0.3,
        diffusion_input: str = "constant",
    ):
        dimensions = (
            (3, hidden_dim, hidden_dim, hidden_dim, 6),   # M1^-1(x)
            (9, hidden_dim, hidden_dim, hidden_dim, 6),   # M2^-1(R)
            (3, hidden_dim, hidden_dim, hidden_dim, 6),   # Dv(v_b): body velocity, not position
            (3, hidden_dim, hidden_dim, hidden_dim, 6),   # Dw(omega_b): body rate, not rotation
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
        self.process_log_sigma = jnp.log(jnp.asarray(initial_process_sigma, dtype=jnp.float32))
        self.log_sigma_observation = jnp.full((4,), jnp.log(jnp.asarray(initial_observation_sigma, dtype=jnp.float32)))
        self.dissipation_enabled = jnp.asarray(
            float(dissipation_enabled), dtype=jnp.float32
        )
        if diffusion_input not in DIFFUSION_INPUTS:
            raise ValueError(f"sde.diffusion_input must be one of {tuple(DIFFUSION_INPUTS)}")
        self.diffusion_input = diffusion_input
        if diffusion_input == "constant":
            self.sigma_net = None
        else:
            # own key stream (fold_in), so the six physical subnetworks initialise exactly as before
            sigma_keys = iter(jax.random.split(jax.random.fold_in(key, 7919), 3))
            net = MLP(sigma_keys, (DIFFUSION_INPUTS[diffusion_input], hidden_dim, hidden_dim, 6), initialization_gain)
            # start at the configured scales: softplus(b) = sigma0  ->  b = log(exp(sigma0) - 1)
            start = jnp.repeat(jnp.asarray(initial_process_sigma, dtype=jnp.float32), 3)
            bias = jnp.log(jnp.expm1(start))
            self.sigma_net = eqx.tree_at(lambda m: m.biases, net, net.biases[:-1] + (bias,))

    def inverse_mass_1(self, position: Array) -> Array:
        return self.M1_net(position)

    def inverse_mass_2(self, rotation_flat: Array) -> Array:
        return self.M2_net(rotation_flat)

    def dissipation_v(self, velocity: Array, position: Array | None = None) -> Array:
        # position is accepted and ignored so the shared report tooling can call every model the same way.
        return self.dissipation_enabled * self.Dv_net(velocity)

    def dissipation_w(self, omega: Array, rotation_flat: Array | None = None) -> Array:
        return self.dissipation_enabled * self.Dw_net(omega)

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

    def diffusion_scale(self, state: Array) -> Array:
        """(6,) diffusion scales. Constant model: twist units [sigma_v x3, sigma_omega x3]. State-dependent
        model: momentum units, a body force (3) and torque (3) scale from sigma_net."""
        if self.sigma_net is None:
            return self.process_sigma()
        return jax.nn.softplus(self.sigma_net(diffusion_features(state, self.diffusion_input)))

    def twist_noise(self, state: Array, noise: Array) -> Array:
        """(6,) twist increment per unit sqrt(dt) for one standard-normal draw ``noise`` (6,)."""
        if self.sigma_net is None:
            return self.process_sigma() * noise
        scale = self.diffusion_scale(state)
        force = self.inverse_mass_1(state[:3]) @ (scale[:3] * noise[:3])
        torque = self.inverse_mass_2(state[3:12]) @ (scale[3:] * noise[3:])
        return jnp.concatenate([force, torque])

    def twist_diffusion_blocks(self, state: Array) -> tuple[Array, Array]:
        """(3,3) linear and angular blocks A_v, A_w of the twist diffusion, so d xi = f dt + blockdiag(A_v, A_w) dW:
        diag(sigma) for the constant model, M^-1(x) diag(sigma(x)) for noise in momentum."""
        if self.sigma_net is not None:
            scale = self.diffusion_scale(state)
            return (self.inverse_mass_1(state[:3]) @ jnp.diag(scale[:3]),
                    self.inverse_mass_2(state[3:12]) @ jnp.diag(scale[3:]))
        sigma = self.process_sigma()
        return jnp.diag(sigma[:3]), jnp.diag(sigma[3:])

    def process_sigma(self) -> Array:
        """Diagonal of the diffusion on the body twist, (6,): [sigma_v x3, sigma_omega x3]."""
        return jnp.repeat(jnp.exp(self.process_log_sigma), 3)

    def observation_sigma(self) -> Array:
        """(4,) observation scales for position, attitude, linear velocity, angular velocity."""
        return jnp.exp(self.log_sigma_observation)

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
            - self.dissipation_v(velocity) @ d_h_dpv
            + force_torque[:3]
        )
        dpw = (
            jnp.cross(momentum_w, d_h_dpw)
            + jnp.cross(momentum_v, d_h_dpv)
            + jnp.cross(rotation, d_h_dr.reshape(3, 3), axis=-1).sum(axis=0)
            - self.dissipation_w(omega) @ d_h_dpw
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
            self.inverse_mass_1(position) @ self.dissipation_v(state[12:15]),
            self.inverse_mass_2(rotation_flat)
            @ self.dissipation_w(state[15:18]),
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
