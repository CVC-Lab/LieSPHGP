r"""Variational random-Fourier-feature GPs on the product manifold $SO(3)^n$.

Self-contained rewrite of ``src/utils/JAX/gp_model.py`` for $n$ links. It lives
here rather than extending that file because the feature tensors change shape
($(F,9)\to(F,n,9)$), which alters the Equinox pytree and would invalidate every
saved single-pendulum checkpoint.

The kernel (§16 of `multi-joint-ph-system.md`)
----------------------------------------------
On $SO(3)$ alone, the geodesic distance to a random base rotation $W_f$ is

.. math:: \theta_f(R)=\arctan2\big(\lVert\mathrm{vee}(\tfrac{M-M^\top}{2})\rVert,\ \tfrac{\mathrm{tr}M-1}{2}\big),\quad M=W_f^\top R .

On the product manifold the natural metric is
$d(q,q')^2=\sum_i\theta(R_i,R_i')^2$, so the Matérn random-Fourier features
become a **sum over links inside the cosine**:

.. math::
    \boxed{\ \varphi_f(q)=\sqrt{\tfrac{2}{F}}\,
        \cos\Big(\sum_{i=1}^{n}\theta_f^{(i)}(R_i)\,\varpi_f^{(i)}+b_f\Big)\ }

with independent base rotations $W_f^{(i)}$ and frequencies $\varpi_f^{(i)}$
per link. At $n=1$ this is exactly the single-pendulum feature map. Summing
*inside* the cosine (rather than adding separate per-link features) is what
makes the kernel a **product** kernel — able to represent the cross-link terms
that $M^{-1}(q)$ needs — instead of a merely additive one.

Numerics
--------
The geodesic angle uses the $\arctan2$ form with the **double-where idiom**
around the `sqrt`: a placeholder inside and the true value outside, so both the
forward pass and the adjoint stay finite at $\theta=0$ (reverse-mode AD would
otherwise propagate $\partial\sqrt x/\partial x$ through the unselected branch
of a plain `jnp.maximum`).
"""
from __future__ import annotations

import math
from typing import Optional

import jax
import jax.numpy as jnp
import equinox as eqx


# ══════════════════════════════════════════════════════════════════════
# Feature maps
# ══════════════════════════════════════════════════════════════════════

class ProductMaternFeatures(eqx.Module):
    r"""RFF approximation of a Matérn kernel on $SO(3)^n$.

    Fields
    ------
    base_rotations : `(F, n, 9)` random $W_f^{(i)}\in SO(3)$, flattened
    omega_angles   : `(F, n)`    Matérn spectral frequencies $\varpi_f^{(i)}$
    phases         : `(F,)`      $b_f\sim\mathcal U[0,2\pi)$
    scale          : `()`        $\sqrt{2/F}$

    Input to `__call__` is a flat `(9n,)` vector
    $(\mathrm{vec}(R_1),\dots,\mathrm{vec}(R_n))$.
    """
    base_rotations: jnp.ndarray
    omega_angles:   jnp.ndarray
    phases:         jnp.ndarray
    scale:          jnp.ndarray
    n_links:    int   = eqx.field(static=True)
    n_features: int   = eqx.field(static=True)
    nu:         float = eqx.field(static=True)
    ell:        float = eqx.field(static=True)

    def __init__(self, key, n_links: int, n_features: int,
                 nu: float = 2.5, ell: float = 1.0, dtype=jnp.float32):
        self.n_links = int(n_links)
        self.n_features = int(n_features)
        self.nu = nu
        self.ell = ell

        k_rot, k_z, k_s, k_p = jax.random.split(key, 4)

        # ── 1. Random base rotations, one per (feature, link) ──
        def rand_rot(k):
            u1, u2, u3 = jax.random.uniform(k, (3,))
            q1 = jnp.sqrt(1 - u1) * jnp.sin(2 * jnp.pi * u2)
            q2 = jnp.sqrt(1 - u1) * jnp.cos(2 * jnp.pi * u2)
            q3 = jnp.sqrt(u1) * jnp.sin(2 * jnp.pi * u3)
            q4 = jnp.sqrt(u1) * jnp.cos(2 * jnp.pi * u3)
            x, y, z, w = q1, q2, q3, q4
            return jnp.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]).flatten()

        n_tot = self.n_features * self.n_links
        self.base_rotations = jax.vmap(rand_rot)(
            jax.random.split(k_rot, n_tot)
        ).reshape(self.n_features, self.n_links, 9).astype(dtype)

        # ── 2. Matérn spectral density as a scale mixture of Gaussians ──
        # $\varpi = z/\sqrt{s/\,2\nu}\,/\,\ell$ with $s\sim\Gamma(\nu,\,\cdot)$
        # is the standard Student-t / Matérn spectrum.
        df = 2.0 * self.nu
        s = jax.random.gamma(k_s, df / 2.0, shape=(self.n_features, 1)) / 0.5
        z = jax.random.normal(k_z, (self.n_features, self.n_links))
        self.omega_angles = (z / jnp.sqrt(s / df) / self.ell).astype(dtype)

        self.phases = (jax.random.uniform(k_p, (self.n_features,))
                       * 2.0 * math.pi).astype(dtype)
        self.scale = jnp.asarray(math.sqrt(2.0 / self.n_features), dtype=dtype)

    def __call__(self, x):
        r"""`x : (9n,)` $\to$ features `(F,)`."""
        R = x.reshape(self.n_links, 9)                       # (n, 9)

        # $\mathrm{tr}(W_f^{(i)\top}R_i)$ is the plain dot product of the
        # flattened matrices.
        traces = jnp.einsum('nj,fnj->fn', R, self.base_rotations)   # (F, n)

        cos_theta = jnp.clip((traces - 1.0) / 2.0, -1.0, 1.0)
        sin_sq = 1.0 - cos_theta * cos_theta
        nonzero = sin_sq > 0.0
        sin_theta = jnp.where(nonzero, jnp.sqrt(jnp.where(nonzero, sin_sq, 1.0)), 0.0)
        angles = jnp.arctan2(sin_theta, cos_theta)           # (F, n)

        # Product kernel: sum the per-link projections *inside* the cosine.
        proj = jnp.sum(angles * self.omega_angles, axis=-1)  # (F,)
        return self.scale * jnp.cos(proj + self.phases)


class PeriodicFeatures(eqx.Module):
    r"""Truncated periodic (Fourier) features on one scalar coordinate.

    Kept for parity with the single-pendulum GP: the full feature map is the
    outer product $\varphi_{\rm matern}\otimes\varphi_{\rm periodic}$, giving
    $D = F\cdot(1+2m_{\max})$ basis functions.
    """
    c0:  jnp.ndarray
    ck:  jnp.ndarray
    m_k: jnp.ndarray
    period: float = eqx.field(static=True)
    ell:    float = eqx.field(static=True)
    m_max:  int   = eqx.field(static=True)

    def __init__(self, period: float = 2.0 * math.pi, ell: float = 1.0,
                 m_max: int = 5, dtype=jnp.float32):
        self.period = period
        self.ell = ell
        self.m_max = m_max

        m = jnp.arange(0, self.m_max + 1)
        a_m = jnp.exp(-2.0 * (math.pi ** 2) * (self.ell ** 2) * (m ** 2)
                      / (self.period ** 2))
        a_m = a_m.at[0].set(1.0)
        a_m = a_m / jnp.sum(a_m)

        self.c0 = jnp.sqrt(a_m[0:1]).astype(dtype)
        self.ck = jnp.sqrt(2.0 * a_m[1:]).astype(dtype)
        self.m_k = jnp.arange(1, self.m_max + 1, dtype=dtype)

    def __call__(self, x_scalar):
        x = x_scalar[..., None]
        args = (2.0 * math.pi / self.period) * x * self.m_k
        harmonics = jnp.stack([self.ck * jnp.cos(args),
                               self.ck * jnp.sin(args)], axis=-1)
        harmonics = harmonics.reshape(*x.shape[:-1], -1)
        return jnp.concatenate([jnp.broadcast_to(self.c0, x.shape), harmonics],
                               axis=-1)


# ══════════════════════════════════════════════════════════════════════
# Variational GP
# ══════════════════════════════════════════════════════════════════════

class GP_NLink(eqx.Module):
    r"""Mean-field variational GP with product-manifold features.

    Weights carry a diagonal Gaussian posterior
    $q_\psi(w)=\mathcal N(\mu,\sigma_w^2)$ against a $\mathcal N(0,I)$ prior.
    Calling with `inference_mode=True` uses the posterior mean; with a `key`
    it draws a reparameterised sample $w=\mu+\sigma_w\epsilon$.
    """
    matern:   ProductMaternFeatures
    periodic: PeriodicFeatures
    w_mean:      jnp.ndarray
    log_w_covar: jnp.ndarray
    Dm:           int   = eqx.field(static=True)
    Dp:           int   = eqx.field(static=True)
    output_dim:   int   = eqx.field(static=True)
    periodic_dim: int   = eqx.field(static=True)

    def __init__(self, key, n_links: int, output_dim: int = 1,
                 n_matern_features: int = 32, nu: float = 2.5,
                 ell_m: float = 1.0, period: float = 2 * math.pi,
                 ell_p: float = 0.5, m_max: int = 5, periodic_dim: int = 0,
                 dtype=jnp.float32):
        k1, k2 = jax.random.split(key)
        self.matern = ProductMaternFeatures(k1, n_links, n_matern_features,
                                            nu, ell_m, dtype=dtype)
        self.periodic = PeriodicFeatures(period, ell_p, m_max, dtype=dtype)
        self.periodic_dim = periodic_dim

        self.Dm = int(n_matern_features)
        self.Dp = 1 + 2 * int(m_max)
        self.output_dim = int(output_dim)

        shape = (self.Dm * self.Dp, self.output_dim)
        # Small-variance init keeps the initial function near zero without the
        # heavy KL floor that a very tight log-sigma would impose.
        self.w_mean = ((1.0 / (self.Dm * self.Dp))
                       * jax.random.normal(k2, shape)).astype(dtype)
        self.log_w_covar = jnp.full(shape, -2.0, dtype=dtype)

    def __call__(self, x, key: Optional[jax.Array] = None,
                 inference_mode: bool = False):
        matern_f = self.matern(x)                             # (Dm,)
        per_f = self.periodic(x[..., self.periodic_dim])      # (Dp,)

        if inference_mode:
            w = self.w_mean
        else:
            if key is None:
                raise ValueError('Key required for training-mode GP sampling')
            # `dtype=` is required, not cosmetic: without it `jax.random.normal`
            # returns float64 whenever `jax_enable_x64` is on (the environment
            # switches it on at import), so the *sampled* weights would promote
            # while the posterior-mean path stayed float32. The mismatch only
            # ever surfaces in training mode, far away, as a primal/tangent
            # dtype error in the `jax.jvp` inside `network.drift`.
            w = self.w_mean + jnp.exp(self.log_w_covar) * jax.random.normal(
                key, self.w_mean.shape, dtype=self.w_mean.dtype)

        w = w.reshape(self.Dm, self.Dp, self.output_dim)
        return jnp.einsum('m,p,mpo->o', matern_f, per_f, w)

    def weight_kl_loss(self):
        r"""$\mathrm{KL}\big(\mathcal N(\mu,\sigma_w^2)\,\Vert\,\mathcal N(0,1)\big)$, summed."""
        var = jnp.exp(2.0 * self.log_w_covar)
        return 0.5 * jnp.sum(var + self.w_mean ** 2 - 1.0 - 2.0 * self.log_w_covar)


class MatrixGP_NLink(eqx.Module):
    r"""GP whose output is reshaped to a matrix of the given `shape`."""
    gp: GP_NLink
    shape: tuple = eqx.field(static=True)

    def __init__(self, key, n_links: int, shape: tuple,
                 n_matern_features: int = 32, dtype=jnp.float32, **kw):
        self.shape = tuple(shape)
        self.gp = GP_NLink(key, n_links=n_links,
                           output_dim=int(shape[0]) * int(shape[1]),
                           n_matern_features=n_matern_features,
                           dtype=dtype, **kw)

    def __call__(self, x, key=None, inference_mode=False):
        return self.gp(x, key=key, inference_mode=inference_mode).reshape(self.shape)

    def weight_kl_loss(self):
        return self.gp.weight_kl_loss()


class PSD_GP_NLink(eqx.Module):
    r"""GP whose output is a positive-semidefinite $d\times d$ matrix, $d=3n$.

    The GP emits the $\tfrac{d(d+1)}{2}$ free entries of a lower-triangular
    Cholesky factor $L$ and returns $LL^\top\succeq0$ — PSD by construction, at
    every input, for every weight sample.

    The diagonal is $L_{ii}=c\tanh\big(\mathrm{softplus}(h_i)/c\big)+\varepsilon$:
    softplus keeps it positive, the $\tanh$ cap at $c$ stops runaway growth
    early in training, and $\varepsilon$ is a conditioning floor. Use
    $\varepsilon=1$ for $M^{-1}$ (which must stay invertible) and
    $\varepsilon=0$ for $D$ (which is allowed to vanish).
    """
    gp: GP_NLink
    diag_dim:     int   = eqx.field(static=True)
    off_diag_dim: int   = eqx.field(static=True)
    epsilon:      float = eqx.field(static=True)
    max_L:        float = eqx.field(static=True)

    def __init__(self, key, n_links: int, diag_dim: int, epsilon: float = 0.0,
                 n_matern_features: int = 32, max_L: float = 1.0,
                 dtype=jnp.float32, **kw):
        self.diag_dim = int(diag_dim)
        self.off_diag_dim = self.diag_dim * (self.diag_dim - 1) // 2
        self.epsilon = float(epsilon)
        self.max_L = float(max_L)
        self.gp = GP_NLink(key, n_links=n_links,
                           output_dim=self.diag_dim + self.off_diag_dim,
                           n_matern_features=n_matern_features,
                           dtype=dtype, **kw)

    def _build(self, vec):
        diag_raw, off_diag = jnp.split(vec, [self.diag_dim], axis=-1)
        diag = (self.max_L * jnp.tanh(jax.nn.softplus(diag_raw) / self.max_L)
                + self.epsilon)

        L = jnp.zeros((self.diag_dim, self.diag_dim), dtype=vec.dtype)
        L = L.at[jnp.diag_indices(self.diag_dim)].set(diag)
        if self.off_diag_dim > 0:
            rows, cols = jnp.tril_indices(self.diag_dim, k=-1)
            L = L.at[rows, cols].set(off_diag)
        return L @ L.T

    def __call__(self, x, key=None, inference_mode=False):
        return self._build(self.gp(x, key=key, inference_mode=inference_mode))

    def weight_kl_loss(self):
        return self.gp.weight_kl_loss()
