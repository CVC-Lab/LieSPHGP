"""PH-NODE losses: the prior work's trajectory loss with RK4 rollouts, and the original real-flight recipe's L1 term.

    rollout_loss   roll the deterministic model out with RK4 from the first (noisy) sample of each window with the
                   recorded controls; L = mean_{t>=1, windows} ( |x_hat - x|^2 + theta(R_hat, R)^2 + |v_hat - v|^2 +
                   |omega_hat - omega|^2 ), theta the geodesic angle; windows whose rollout became non-finite are left
                   out of the mean.
    l1_structure   the original PH-NODE real-flight code's sparsity term (LieGroupHamDL train_quadrotor_SE3_PX4.py:205-216)
                   at the first sample of each window: mean |G(R, x)| + mean |D_v(v)| + mean |D_omega(omega)|. It carries
                   no physical values.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from lie_ph.integrator import log_so3

from .integrator import rollout

Array = jax.Array
NOISE_DIMENSION = 6


def rollout_loss(model, windows: Array, interval: float, substeps: int) -> Array:
    controls = jnp.swapaxes(windows[1:, :, 18:], 0, 1)                                      # (B, T-1, n_u)
    noise = jnp.zeros((windows.shape[0] - 1, substeps, NOISE_DIMENSION), windows.dtype)
    predicted = jax.vmap(lambda x0, u: rollout(model, x0, u, interval, noise))(windows[0], controls)   # (B, T, 18 + n_u)
    observed = jnp.swapaxes(windows, 0, 1)
    angle = jax.vmap(jax.vmap(lambda a, b: log_so3(b[3:12].reshape(3, 3).T @ a[3:12].reshape(3, 3))))(
        predicted[:, 1:], observed[:, 1:])
    error = (jnp.sum(jnp.square(predicted[:, 1:, :3] - observed[:, 1:, :3]), axis=-1) + jnp.sum(angle * angle, axis=-1)
             + jnp.sum(jnp.square(predicted[:, 1:, 12:18] - observed[:, 1:, 12:18]), axis=-1))
    per_window = jnp.mean(error, axis=1)
    healthy = jax.lax.stop_gradient(jnp.isfinite(per_window))
    safe = jnp.where(healthy, per_window, 0.0)
    return jnp.sum(safe) / jnp.maximum(jnp.sum(healthy), 1.0)


def l1_structure(model, windows: Array) -> Array:
    first = windows[0]
    control = jax.vmap(lambda s: model.control_matrix(s[3:12], s[:3]))(first)
    linear = jax.vmap(lambda s: model.dissipation_v(s[12:15]))(first)
    angular = jax.vmap(lambda s: model.dissipation_w(s[15:18]))(first)
    return jnp.mean(jnp.abs(control)) + jnp.mean(jnp.abs(linear)) + jnp.mean(jnp.abs(angular))
