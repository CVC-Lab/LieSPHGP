r"""Stratonovich Lie–Heun integrator on $SO(3)^n\times\mathbb{R}^{3n}$ in $(q,p)$ form.

Self-contained (including the $SO(3)$ helpers) so nothing outside
``src/models/arm_n_link/`` is imported or modified.

This is the **model-side** integrator: it drives a learned network through the
same geometry as the ground-truth environment
(``envs/arm_nlink_so3/arm_nlink_physics.py::lie_heun_step``). The model is
expected to expose

    drift_p(q, p, u)                -> $\dot p$          $\in\mathbb{R}^{3n}$
    stochastic_increment_p(q, dW)   -> $\Sigma_\theta(q)dW$ $\in\mathbb{R}^{3n}$
    M_inv(q)                        -> $M^{-1}(q)$       $\in\mathbb{R}^{3n\times3n}$
    n                               -> number of links (static int)

Conventions
-----------
    q      : (9n,)   $(\mathrm{vec}(R_1),\dots,\mathrm{vec}(R_n))$, row-major
    p      : (3n,)   momentum; block $i$ is `p[3i:3i+3]`
    x_qp   : (12n,)  `concat(q, p)` — the internal scan state
    x0     : (12n,)  `concat(q, omega)` — the external rollout convention
    dW     : (3,)    **3-dimensional**, already scaled by $\sqrt h$

The Wiener increment is 3-D, not $3n$-D: physically there is a single world
wind field, so the diffusion $\Sigma(q)\in\mathbb{R}^{3n\times3}$ is low rank
(§9.3 of `multi-joint-ph-system.md`).

Externally the rollout takes and emits $\omega$ (matching the dataset), while
the scan carries $p$; conversion happens only at the boundaries via
$p = M(q)\omega = \mathrm{solve}(M^{-1}(q),\omega)$ and $\omega = M^{-1}(q)p$.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


# ══════════════════════════════════════════════════════════════════════
# SO(3) helpers
# ══════════════════════════════════════════════════════════════════════

def hat(w):
    r"""Hat map $[\cdot]_\times:\mathbb{R}^3\to\mathfrak{so}(3)$."""
    return jnp.array([
        [0.0, -w[2], w[1]],
        [w[2], 0.0, -w[0]],
        [-w[1], w[0], 0.0],
    ], dtype=w.dtype)


def exp_so3(phi):
    r"""Rodrigues' formula $\exp([\phi]_\times)$, gradient-safe at $\phi=0$.

    .. math::
        \exp([\phi]_\times)=I+\frac{\sin\theta}{\theta}[\phi]_\times
            +\frac{1-\cos\theta}{\theta^2}[\phi]_\times^2,\qquad \theta=\lVert\phi\rVert

    The `jnp.where` + safe-divide pattern substitutes $\theta=1$ in the branch
    that is not taken, so neither the forward pass nor the adjoint ever divides
    by zero.
    """
    theta_sq = jnp.dot(phi, phi)
    small = theta_sq < 1e-20
    theta_safe = jnp.sqrt(jnp.where(small, 1.0, theta_sq))

    A = jnp.where(small, 1.0 - theta_sq / 6.0, jnp.sin(theta_safe) / theta_safe)
    B = jnp.where(small, 0.5 - theta_sq / 24.0,
                  (1.0 - jnp.cos(theta_safe)) / (theta_safe * theta_safe))

    Phi = hat(phi)
    return jnp.eye(3, dtype=phi.dtype) + A * Phi + B * (Phi @ Phi)


exp_so3_batch = jax.vmap(exp_so3)          # (n, 3) -> (n, 3, 3)


# ══════════════════════════════════════════════════════════════════════
# Single substep
# ══════════════════════════════════════════════════════════════════════

def lie_heun_sde_step_nlink(model, x_qp, u, h, dW):
    r"""One Stratonovich Heun substep on $SO(3)^n\times\mathbb{R}^{3n}$.

    .. math::
        \textbf{(1)}\;& \xi^{(1)}=M^{-1}(q)p,\quad \dot p^{(1)},\quad
            \Delta p^{(1)}=\Sigma_\theta(q)dW,\quad \phi^{(1)}_i=\xi^{(1)}_i h\\
        \textbf{(2)}\;& R^{\rm pr}_i=R_i\exp([\phi^{(1)}_i]_\times),\quad
            p^{\rm pr}=p+\dot p^{(1)}h+\Delta p^{(1)}\\
        \textbf{(3)}\;& \text{re-evaluate at }(q^{\rm pr},p^{\rm pr})
            \text{ reusing the \emph{same} }dW\\
        \textbf{(4)}\;& \bar\phi_i=\tfrac12(\phi^{(1)}_i+\phi^{(2)}_i),\quad
            R^{+}_i=R_i\exp([\bar\phi_i]_\times),\\
        & p^{+}=p+\tfrac h2(\dot p^{(1)}+\dot p^{(2)})
                 +\tfrac12(\Delta p^{(1)}+\Delta p^{(2)})

    Averaging happens in the Lie **algebra**, followed by one exponential per
    link, so every $R_i$ stays on $SO(3)$ by construction. Reusing `dW` across
    both stages is what makes the scheme Stratonovich rather than Itô.
    """
    n = model.n
    q = x_qp[:9 * n]
    p = x_qp[9 * n:]
    R = q.reshape(n, 3, 3)

    # ── Stage 1 ──
    p_dot_1 = model.drift_p(q, p, u)
    omega_1 = model.M_inv(q) @ p
    dp_stoch_1 = model.stochastic_increment_p(q, dW)
    phi_1 = omega_1.reshape(n, 3) * h

    # ── Stage 2: predictor ──
    R_pred = R @ exp_so3_batch(phi_1)
    q_pred = R_pred.reshape(9 * n)
    p_pred = p + p_dot_1 * h + dp_stoch_1

    # ── Stage 3: re-evaluate ──
    p_dot_2 = model.drift_p(q_pred, p_pred, u)
    omega_2 = model.M_inv(q_pred) @ p_pred
    dp_stoch_2 = model.stochastic_increment_p(q_pred, dW)
    phi_2 = omega_2.reshape(n, 3) * h

    # ── Stage 4: corrector ──
    phi_avg = 0.5 * (phi_1 + phi_2)
    q_new = (R @ exp_so3_batch(phi_avg)).reshape(9 * n)
    p_new = p + 0.5 * (p_dot_1 + p_dot_2) * h + 0.5 * (dp_stoch_1 + dp_stoch_2)

    return jnp.concatenate([q_new, p_new])


def lie_heun_ode_step_nlink(model, x_qp, u, h):
    """Deterministic variant — identical geometry with the diffusion absent."""
    n = model.n
    zero_dW = jnp.zeros((3,), dtype=x_qp.dtype)
    del zero_dW  # the ODE path must not touch `stochastic_increment_p` at all

    q = x_qp[:9 * n]
    p = x_qp[9 * n:]
    R = q.reshape(n, 3, 3)

    p_dot_1 = model.drift_p(q, p, u)
    phi_1 = (model.M_inv(q) @ p).reshape(n, 3) * h

    R_pred = R @ exp_so3_batch(phi_1)
    q_pred = R_pred.reshape(9 * n)
    p_pred = p + p_dot_1 * h

    p_dot_2 = model.drift_p(q_pred, p_pred, u)
    phi_2 = (model.M_inv(q_pred) @ p_pred).reshape(n, 3) * h

    q_new = (R @ exp_so3_batch(0.5 * (phi_1 + phi_2))).reshape(9 * n)
    p_new = p + 0.5 * (p_dot_1 + p_dot_2) * h
    return jnp.concatenate([q_new, p_new])


# ══════════════════════════════════════════════════════════════════════
# Rollouts
# ══════════════════════════════════════════════════════════════════════

def _to_p(model, q, omega):
    r"""$p = M(q)\,\omega$; `M_net` returns $M^{-1}$, so this is a solve."""
    return jnp.linalg.solve(model.M_inv(q), omega)


def lie_heun_sde_rollout_nlink(model, x0, u, h, dW_per_outer):
    r"""Roll the SDE out, emitting the state at every outer-step boundary.

    Args:
        x0           : `(12n,)` initial `concat(q, omega)`; `q` must lie on $SO(3)^n$.
        u            : `(u_dim,)` constant, or `(n_outer, u_dim)` one row per
                       outer step. Held constant across the substeps of an
                       outer step, matching the environment's `step()`.
        h            : substep size; outer step $=h\cdot n_{\rm sub}$.
        dW_per_outer : `(n_outer, n_substeps, 3)`, already scaled by $\sqrt h$.

    Returns:
        `(n_outer + 1, 12n)` in `(q, omega)` form, with `x0` prepended.
    """
    n = model.n
    q0 = x0[:9 * n]
    omega0 = x0[9 * n:]
    x0_qp = jnp.concatenate([q0, _to_p(model, q0, omega0)])

    n_outer = dW_per_outer.shape[0]
    u = jnp.asarray(u)
    u_per_outer = (jnp.broadcast_to(u[None, :], (n_outer,) + u.shape)
                   if u.ndim == 1 else u)

    def outer_step(x_qp, scan_in):
        u_t, dW_outer = scan_in

        def inner_step(carry, dW):
            return lie_heun_sde_step_nlink(model, carry, u_t, h, dW), None

        x_qp_new, _ = jax.lax.scan(inner_step, x_qp, dW_outer)
        q_new = x_qp_new[:9 * n]
        omega_new = model.M_inv(q_new) @ x_qp_new[9 * n:]
        return x_qp_new, jnp.concatenate([q_new, omega_new])

    _, x_outer = jax.lax.scan(outer_step, x0_qp, (u_per_outer, dW_per_outer))
    return jnp.concatenate([x0[None], x_outer], axis=0)


def lie_heun_ode_rollout_nlink(model, x0, u, h, n_substeps, n_outer):
    """Deterministic rollout. Same I/O as the SDE variant with `dW` absent."""
    n = model.n
    q0 = x0[:9 * n]
    x0_qp = jnp.concatenate([q0, _to_p(model, q0, x0[9 * n:])])

    u = jnp.asarray(u)
    u_per_outer = (jnp.broadcast_to(u[None, :], (n_outer,) + u.shape)
                   if u.ndim == 1 else u)

    def outer_step(x_qp, u_t):
        def inner_step(carry, _):
            return lie_heun_ode_step_nlink(model, carry, u_t, h), None

        x_qp_new, _ = jax.lax.scan(inner_step, x_qp, jnp.arange(n_substeps))
        q_new = x_qp_new[:9 * n]
        omega_new = model.M_inv(q_new) @ x_qp_new[9 * n:]
        return x_qp_new, jnp.concatenate([q_new, omega_new])

    _, x_outer = jax.lax.scan(outer_step, x0_qp, u_per_outer)
    return jnp.concatenate([x0[None], x_outer], axis=0)
