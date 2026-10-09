"""PH-NODE trajectory losses: roll the deterministic model out with RK4 from the first (noisy) sample of each window.

    rollout            L = mean_{t>=1, windows} ( theta(R_hat_t, R_t)^2 + |omega_hat_t - omega_t|^2 ),  theta the
                       geodesic angle; windows whose rollout became non-finite are left out of the mean (PH-NODE).
    rollout_reference  rotmat_L2_geodesic_loss of Duong & Atanasov (port_ham_neural_ode, utils.py) (PH-NODE-ref):
                       L = mean( |[omega_hat, u_hat] - [omega, u]|^2 over the 6 components ) + mean( theta(GS(R_hat), R)^2 ),
                       with Gram-Schmidt re-orthonormalisation of the predicted rotation, u_hat = u exactly (those three
                       terms are zero but count in the mean, as in their code with u_dim = 3), no masking.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from lie_ph.integrator import log_so3

from .integrator import rollout

Array = jax.Array
NOISE_DIMENSION = 3


def _predict(model, windows: Array, interval: float, substeps: int) -> tuple[Array, Array]:
    """(predicted (B, T, 15), observed (B, T, 15)) from time-major (T, B, 15) windows."""
    controls = jnp.swapaxes(windows[1:, :, 12:15], 0, 1)                                     # (B, T-1, 3)
    noise = jnp.zeros((windows.shape[0] - 1, substeps, NOISE_DIMENSION), windows.dtype)
    predicted = jax.vmap(lambda x0, u: rollout(model, x0, u, interval, noise))(windows[0], controls)   # (B, T, 15)
    return predicted, jnp.swapaxes(windows, 0, 1)


def rollout_loss(model, windows: Array, interval: float, substeps: int) -> Array:
    predicted, observed = _predict(model, windows, interval, substeps)
    angle = jax.vmap(jax.vmap(lambda a, b: log_so3(b[:9].reshape(3, 3).T @ a[:9].reshape(3, 3))))(predicted[:, 1:], observed[:, 1:])
    error = jnp.sum(angle * angle, axis=-1) + jnp.sum(jnp.square(predicted[:, 1:, 9:12] - observed[:, 1:, 9:12]), axis=-1)
    per_window = jnp.mean(error, axis=1)
    healthy = jax.lax.stop_gradient(jnp.isfinite(per_window))
    safe = jnp.where(healthy, per_window, 0.0)
    return jnp.sum(safe) / jnp.maximum(jnp.sum(healthy), 1.0)


def gram_schmidt_rows(rotation_flat: Array) -> Array:
    """Duong & Atanasov's compute_rotation_matrix_from_unnormalized_rotmat on one row-major 9-vector:
    x = row0 / |row0|, z = (x x row1) / |.|, y = z x x; returns the 3 x 3 matrix with rows x, y, z."""
    def unit(v):
        return v / jnp.maximum(jnp.linalg.norm(v), 1.0e-8)
    x = unit(rotation_flat[0:3])
    z = unit(jnp.cross(x, rotation_flat[3:6]))
    return jnp.stack([x, jnp.cross(z, x), z])


def rollout_loss_reference(model, windows: Array, interval: float, substeps: int) -> Array:
    predicted, observed = _predict(model, windows, interval, substeps)
    angle = jax.vmap(jax.vmap(lambda a, b: log_so3(b[:9].reshape(3, 3).T @ gram_schmidt_rows(a[:9]))))(predicted[:, 1:], observed[:, 1:])
    geodesic = jnp.mean(jnp.sum(angle * angle, axis=-1))
    l2 = jnp.mean(jnp.square(predicted[:, 1:, 9:15] - observed[:, 1:, 9:15]))                # omega and (zero) u error
    return l2 + geodesic
