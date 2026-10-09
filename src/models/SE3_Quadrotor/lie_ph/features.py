"""Smooth random Fourier features of a Matern kernel on the chordal embedding of SO(3) (and on R^n).

    phi(x) = sqrt(2 / n) cos(Omega x / ell + b),   Omega_k = z_k / sqrt(s_k / (2 nu)),  z_k ~ N(0, I), s_k ~ chi^2_{2 nu},
    b_k ~ U(0, 2 pi),

the standard Student-t spectral sample of a Matern-nu kernel k(|x - x'| / ell). A rotation enters as vec(R), so the
kernel depends on the chordal distance |R - R'|_F = 2 sqrt(2) sin(theta / 2) (theta = geodesic angle): a valid
kernel on SO(3) (restriction of a positive-definite kernel on R^9) that is infinitely differentiable in R.

Why not the geodesic-angle features of src/utils/JAX/gp_model.py: there the feature depends on theta_k(R), the
geodesic distance to a random base rotation, which has a cone at the base rotation, so V(R) has kinks and its
Hessian jumps. The EKF differentiates through the Jacobian of the dynamics (the Hessian of V) once more for the
training gradient; with the angle features that gradient was dominated by those spikes (float32 gradient
direction cosine 0.01 against float64). These features have bounded derivatives of every order.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

Array = jax.Array


def initialize_features(key: Array, input_dimension: int, count: int, smoothness: float,
                        length_scale: float) -> dict[str, Array]:
    key_normal, key_gamma, key_phase = jax.random.split(key, 3)
    degrees = 2.0 * smoothness
    # float32 draws: the same features whether or not jax_enable_x64 is on
    chi_square = 2.0 * jax.random.gamma(key_gamma, degrees / 2.0, shape=(count,), dtype=jnp.float32)
    frequencies = (jax.random.normal(key_normal, (count, input_dimension), jnp.float32)
                   / jnp.sqrt(chi_square[:, None] / degrees))
    return {"frequencies": frequencies / length_scale,
            "phases": jax.random.uniform(key_phase, (count,), jnp.float32, maxval=2.0 * math.pi)}


def features(setup: dict[str, Array], value: Array) -> Array:
    count = setup["phases"].shape[0]
    return math.sqrt(2.0 / count) * jnp.cos(setup["frequencies"].astype(value.dtype) @ value
                                            + setup["phases"].astype(value.dtype))
