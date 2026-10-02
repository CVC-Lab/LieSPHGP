r"""ELBO components for the $n$-link variational-GP port-Hamiltonian SDE.

Self-contained: geodesic distance, rollout NLL, per-increment pseudo-likelihood
and per-subnet KL, all generalized from 1 link to $n$.

    $-\mathrm{ELBO} = L_{\rm NLL} + \lambda_{\rm PL}L_{\rm PL} + \tfrac{\beta}{N}L_{\rm KL}$

Data layout: the last axis of a trajectory is $15n$, split as
$(9n\ \text{rotations},\ 3n\ \text{rates},\ 3n\ \text{torques})$.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp

_LOG_2PI = math.log(2.0 * math.pi)


# ══════════════════════════════════════════════════════════════════════
# Geodesic distance on SO(3)
# ══════════════════════════════════════════════════════════════════════

def geodesic_distance(m1, m2):
    r"""$\theta\in[0,\pi]$ between batched rotations. `m1, m2 : (B, 3, 3)`.

    .. math:: \theta=\arctan2\big(\lVert\mathrm{vee}(\tfrac{M-M^\top}{2})\rVert,\ \tfrac{\mathrm{tr}M-1}{2}\big),\quad M=R_1R_2^\top

    The two-argument form needs no clipping and keeps full precision for small
    $\theta$, unlike $\arccos(\tfrac{\mathrm{tr}M-1}{2})$ which loses ~6 digits
    near identity. The double-`where` around the `sqrt` keeps the adjoint finite
    at $\theta=0$, where the correct subgradient of $\theta^2$ is zero.
    """
    M = jnp.matmul(m1, jnp.transpose(m2, (0, 2, 1)))
    cos_theta = (M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2] - 1.0) / 2.0

    sin_axis = jnp.stack([
        (M[:, 2, 1] - M[:, 1, 2]) * 0.5,
        (M[:, 0, 2] - M[:, 2, 0]) * 0.5,
        (M[:, 1, 0] - M[:, 0, 1]) * 0.5,
    ], axis=-1)

    sin_sq = jnp.sum(sin_axis * sin_axis, axis=-1)
    nonzero = sin_sq > 0.0
    sin_theta = jnp.where(nonzero, jnp.sqrt(jnp.where(nonzero, sin_sq, 1.0)), 0.0)
    return jnp.arctan2(sin_theta, cos_theta)


# ══════════════════════════════════════════════════════════════════════
# Rollout likelihood
# ══════════════════════════════════════════════════════════════════════

def elbo_nll_nlink(traj_obs, traj_hat, log_sigma_R, log_sigma_omega, n):
    r"""Rollout negative log-likelihood over `(..., 15n)` tensors.

    Rotations use the concentrated Gaussian on $SO(3)$, **summed over links**:

    .. math::
        -\log p(\tilde q\mid\hat q)=\sum_{i=1}^{n}
            \Big[\tfrac{\theta_i^2}{2\sigma_R^2}+3\log\sigma_R+\tfrac32\log2\pi\Big]

    The 3 is the *manifold* dimension of $SO(3)$, not the dimension of the
    scalar $\theta$ — the likelihood is over $R$, so the partition function
    $Z\approx(2\pi\sigma_R^2)^{3/2}$ picks it up. With $n$ links the whole
    bracket appears $n$ times.

    Rates are a plain $3n$-dimensional isotropic Gaussian:

    .. math::
        -\log p(\tilde\omega\mid\hat\omega)
            =\tfrac{\lVert\Delta\omega\rVert^2}{2\sigma_\omega^2}
             +3n\log\sigma_\omega+\tfrac{3n}{2}\log2\pi

    The $n$-scaling of both normalisers matters: it fixes the relative weight of
    the rotation head, the rate head and the KL term. Torques are sliced off —
    they are inputs, not predictions.
    """
    rd, ad = 9 * n, 3 * n

    R_obs = traj_obs[..., :rd].reshape(-1, 3, 3)
    R_hat = traj_hat[..., :rd].reshape(-1, 3, 3)
    w_obs = traj_obs[..., rd:rd + ad].reshape(-1, ad)
    w_hat = traj_hat[..., rd:rd + ad].reshape(-1, ad)

    # ── rotations: mean over (samples x links), then x n for the per-sample total
    theta_sq = geodesic_distance(R_obs, R_hat) ** 2
    mean_theta_sq_per_link = jnp.mean(theta_sq)
    sigma_R_sq = jnp.exp(2.0 * log_sigma_R)
    nll_R = n * (0.5 * mean_theta_sq_per_link / sigma_R_sq
                 + 3.0 * log_sigma_R + 1.5 * _LOG_2PI)

    # ── rates: sum over the 3n components, then mean over samples
    diff = w_obs - w_hat
    mean_omega_sq = jnp.mean(jnp.sum(diff * diff, axis=-1))
    sigma_w_sq = jnp.exp(2.0 * log_sigma_omega)
    nll_omega = (0.5 * mean_omega_sq / sigma_w_sq
                 + 3.0 * n * log_sigma_omega + 1.5 * n * _LOG_2PI)

    return {
        'nll_total':     nll_R + nll_omega,
        'nll_R':         nll_R,
        'nll_omega':     nll_omega,
        'mean_theta_sq': mean_theta_sq_per_link,   # per link — comparable across n
        'mean_omega_sq': mean_omega_sq,
        'sigma_R':       jnp.exp(log_sigma_R),
        'sigma_omega':   jnp.exp(log_sigma_omega),
    }


# ══════════════════════════════════════════════════════════════════════
# Per-increment pseudo-likelihood (Woodbury)
# ══════════════════════════════════════════════════════════════════════

def pl_loss_nlink(model, batch_x_cat, dt, sigma_obs_omega, gp_keys_batch,
                  n, inference_mode=False, floor=1e-6):
    r"""Euler–Maruyama transition density at consecutive observed snapshots.

    The rollout NLL gives the diffusion no usable gradient — model and
    environment have independent Brownian paths, so enlarging $\Sigma_\theta$
    only adds variance to the residual and the optimum is $\Sigma_\theta\to0$.
    This term instead evaluates

    .. math::
        \Delta\omega_{\rm obs}\ \sim\ \mathcal N\big(\mu(q_t,\omega_t,u_t)\,\Delta t,\ \ \Sigma_{\rm eff}\big),
        \qquad
        \Sigma_{\rm eff}=\Delta t\,AA^\top+\underbrace{2\sigma_{\rm obs}^2}_{=:s} I_{3n}

    where

    .. math::
        A(q) \;:=\; M_\theta^{-1}(q)\,\Sigma_\theta(q) \ \in\ \mathbb{R}^{3n\times3}

    **is the $\omega$-space diffusion, not $\Sigma$ itself.** $\Sigma_\theta$ is
    defined in *momentum* space — the integrator adds $\Sigma\,dW$ to $p$ — so
    the change of variables $\omega = M^{-1}(q)p$ carries it through as

    .. math::
        d\omega \;=\; \underbrace{\big[M^{-1}f+\dot{\overline{(M^{-1})}}p\big]}_{=\;\mu}dt
                 \;+\; M^{-1}\Sigma\,dW

    exactly the same transformation `model.drift` already applies to the drift.
    Using $\Sigma$ in place of $A$ is nearly harmless when $M^{-1}\approx I$, but
    it decouples the diffusion from the drift's scale gauge
    $(M^{-1},V,D,g,\Sigma)\to(\beta M^{-1},\tfrac1\beta(\cdot))$: nothing then
    ties $\Sigma$ to the mass scale, $\beta$ drifts freely, and the fitted
    $\Sigma$ ends up inconsistent with the rest of the model by that factor.

    The factor 2 in $s$ is because $\Delta\omega_{\rm obs}$ differences two
    independently-noisy $\omega$ measurements.

    **Why Woodbury.** $A\in\mathbb{R}^{3n\times3}$, so $AA^\top$ has rank 3 in
    $3n$ dimensions — singular on its own. The observation-noise floor is what
    makes $\Sigma_{\rm eff}$ invertible, and exploiting the
    low-rank-plus-scaled-identity structure keeps the cost $O(n)$ rather than
    $O(n^3)$:

    .. math::
        r^\top\Sigma_{\rm eff}^{-1}r
            &=\tfrac1s\Big[\lVert r\rVert^2-(A^\top r)^\top K^{-1}(A^\top r)\Big],
            \qquad K=\tfrac{s}{\Delta t}I_3+A^\top A \\
        \log\det\Sigma_{\rm eff}&=3n\log s+\log\det\big(I_3+\tfrac{\Delta t}{s}A^\top A\big)

    Both the inverse and the determinant reduce to $3\times3$ problems.

    Args:
        sigma_obs_omega : frozen at the dataset's `obs_noise_std`. Not learnable
            — as a free parameter the optimiser absorbs model bias into it and
            masks errors in `V_net` / `Dw_net`.
        floor : lower bound on $s$. Required: with `obs_noise_std = 0` the
            matrix above is genuinely singular and the loss would be $-\infty$.
    """
    T_obs, B, _ = batch_x_cat.shape
    S = next(iter(gp_keys_batch.values())).shape[1]
    rd, ad = 9 * n, 3 * n
    Tm1 = T_obs - 1

    q_t = batch_x_cat[:-1, :, :rd]
    omega_t = batch_x_cat[:-1, :, rd:rd + ad]
    # Row k stores the torque that produced obs_k (the generator appends
    # (obs, curr_u) before advancing curr_u), so the torque acting on the
    # transition t -> t+1 lives at row t+1, not row t. Using [:-1] here lags
    # the whole sequence by one step whenever `random_u` is on — see the note
    # in train.py::rollout_single.
    u_t = batch_x_cat[1:, :, rd + ad:rd + 2 * ad]
    delta_om = batch_x_cat[1:, :, rd:rd + ad] - omega_t

    q_tbs = jnp.broadcast_to(q_t[:, :, None, :], (Tm1, B, S, rd))
    w_tbs = jnp.broadcast_to(omega_t[:, :, None, :], (Tm1, B, S, ad))
    u_tbs = jnp.broadcast_to(u_t[:, :, None, :], (Tm1, B, S, ad))
    d_tbs = jnp.broadcast_to(delta_om[:, :, None, :], (Tm1, B, S, ad))
    keys_tbs = {k: jnp.broadcast_to(v[None], (Tm1, B, S, 2))
                for k, v in gp_keys_batch.items()}

    s = jnp.maximum(2.0 * float(sigma_obs_omega) ** 2, floor)
    dt_f = jnp.asarray(dt, dtype=q_tbs.dtype)

    def per_step(q, om, u, dom, keys):
        eff_keys = None if inference_mode else keys
        mu = model.drift(q, om, u, keys=eff_keys)                # (3n,)
        Sigma = model.Sigma(q, keys=eff_keys)                    # (3n, 3), p-space

        # $\Sigma_\theta$ is the **momentum**-space diffusion: the integrator adds
        # $\Sigma dW$ to $p$, not to $\omega$. This density is over $\Delta\omega$,
        # so the diffusion must be pushed through the same change of variables
        # that produced `mu`:  $d\omega = \mu\,dt + M^{-1}\Sigma\,dW$.
        # Using $\Sigma$ here instead of $A=M^{-1}\Sigma$ silently fits the wrong
        # matrix; it looks benign whenever $M^{-1}\approx I$ but decouples the
        # diffusion from the drift's scale gauge, letting $\beta$ run away.
        A = model.M_inv(q, keys=eff_keys) @ Sigma                # (3n, 3)

        r = dom - mu * dt_f                                      # (3n,)
        AtA = A.T @ A                                            # (3, 3)
        K = (s / dt_f) * jnp.eye(3, dtype=q.dtype) + AtA
        Ar = A.T @ r                                             # (3,)

        quad = (jnp.sum(r * r) - jnp.dot(Ar, jnp.linalg.solve(K, Ar))) / s
        _, logdet3 = jnp.linalg.slogdet(
            jnp.eye(3, dtype=q.dtype) + (dt_f / s) * AtA)
        logdet = 3.0 * n * jnp.log(s) + logdet3

        nll = 0.5 * quad + 0.5 * logdet
        # Reported readout stays $\lVert\Sigma\rVert_F$ (p-space), so the training
        # log column remains directly comparable to the analytic wind map and to
        # every run made before this fix. `omega_fro` is the quantity the
        # likelihood actually sees.
        sigma_fro = jnp.sqrt(jnp.sum(Sigma * Sigma))
        omega_fro = jnp.sqrt(jnp.sum(A * A))
        return nll, jnp.sum(r * r), sigma_fro, omega_fro

    nll_arr, rsq_arr, sig_arr, om_arr = jax.vmap(jax.vmap(jax.vmap(per_step)))(
        q_tbs, w_tbs, u_tbs, d_tbs, keys_tbs)

    return {
        'pl_loss':          jnp.mean(nll_arr),
        'mean_residual_sq': jnp.mean(rsq_arr),
        'mean_sigma_fro':   jnp.mean(sig_arr),
        'mean_omega_fro':   jnp.mean(om_arr),
        'sigma_obs_omega':  jnp.asarray(float(sigma_obs_omega)),
    }


# ══════════════════════════════════════════════════════════════════════
# KL
# ══════════════════════════════════════════════════════════════════════

def kl_per_subnet(model):
    """Closed-form mean-field KL for each GP subnet, plus the total.

    A subnet may be `None` (the ODE variants have no `Sigma_net`); its KL is
    zero — absent by construction, like the NN modules' zero KL.
    """
    out = {}
    for name in ('M', 'V', 'Dw', 'g', 'Sigma'):
        net = getattr(model, f'{name}_net')
        out[f'{name}_kl'] = (jnp.zeros(()) if net is None
                             else net.weight_kl_loss())
    out['total_kl'] = sum(out.values())
    return out
