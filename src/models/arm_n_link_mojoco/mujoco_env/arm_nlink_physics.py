r"""Pure-JAX physics of the $n$-link windy robotic arm on $SO(3)^n$.

This module is the **ground-truth port-Hamiltonian system** described in
`multi-joint-ph-system.md`. It contains no Gym machinery and no mutable state —
every function is a pure, jittable function of `(params, state)`, so each formula
in the write-up can be tested on its own (see `mini_tests/test_gt_pH_matches_arm_env.py`).

Configuration manifold (§2 of the write-up)
-------------------------------------------
$n$ links in a serial chain joined by **ball joints** (3 rotational DOF each).
Each link carries its **absolute (world-frame) attitude** $R_i \in SO(3)$:

.. math::   q = (R_1,\dots,R_n)\in G = SO(3)^n, \qquad \dim G = 3n .

Velocities are left-trivialized (body-frame) angular rates,
$\dot R_i = R_i[\omega_i]_\times$, and the conjugate momentum is $p = M(q)\omega$.

The system (§6, §10)
--------------------
With $\xi := \partial H/\partial p = M^{-1}(q)p$ and
$H(q,p) = \tfrac12 p^\top M^{-1}(q)p + V(q)$:

.. math::

    dR_i &= R_i[\xi_i]_\times\,dt \\
    dp   &= \big(\hat P\xi + \mathcal T(H) - D(q,\omega)\xi + g(q)u
              + \Sigma(q)\,w(t)\mathrm d\big)dt \;+\; \sigma\,\Sigma(q)\circ dW_t

where $\hat P = \mathrm{blkdiag}([p_i]_\times)$ is the gyroscopic ($\mathrm{ad}^*$)
term, $\mathcal T$ is the trivialized gradient operator, $D$ is the dissipation
matrix, $g(q)$ the input map and $\Sigma(q)$ the wind lever map.

Array-shape conventions (kept uniform everywhere)
-------------------------------------------------
    R        : (n, 3, 3)     row-major, `R[i]` is link $i$'s world attitude
    omega, p : (n, 3)        per-link body-frame rate / momentum
    q_flat   : (9n,)         `R.reshape(-1)` — the redundant embedding
    u        : (n, 3)        actuator torque at joint $i$, link-$i$ body frame
    M, D     : (3n, 3n)
    T, G     : (3n, 3n)
    Sigma    : (3n, 3)       low rank: one shared 3-D wind field

Stacked $3n$-vectors are always `x.reshape(-1)` of an `(n, 3)` array, so block
$i$ of a $3n$-vector is `x[i]`. Never mix the two orderings.
"""
from __future__ import annotations

import os
import sys
from typing import NamedTuple

import jax
import jax.numpy as jnp

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# This file is a verbatim copy of envs/arm_nlink_SO3/arm_nlink_physics.py, moved
# two levels deeper (src/models/arm_n_link_mojoco/mujoco_env/), so reaching the
# project root -- which is what makes `src.utils.JAX.ode_utils_jax` importable
# -- takes four hops instead of two. This is the ONLY edit relative to the
# original; keep it that way so the two files stay diffable.
PROJECT_ROOT = os.path.abspath(
    os.path.join(THIS_FILE_DIR, '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Reuse the project's canonical, gradient-safe SO(3) helpers rather than
# re-deriving them here (they already handle the small-angle Taylor branch).
from src.utils.JAX.ode_utils_jax import hat, exp_so3   # noqa: E402

# Batched versions: apply the single-element map to every link.
hat_batch = jax.vmap(hat)              # (n, 3)    -> (n, 3, 3)
exp_so3_batch = jax.vmap(exp_so3)      # (n, 3)    -> (n, 3, 3)

E_Z = jnp.array([0.0, 0.0, 1.0])


# ══════════════════════════════════════════════════════════════════════
# Parameters
# ══════════════════════════════════════════════════════════════════════

class ArmParams(NamedTuple):
    r"""Physical parameters of the chain. A JAX pytree (NamedTuple), so it can
    be passed straight through `jit` / `grad` / `vmap`.

    All per-link quantities have leading dimension $n$; the number of links is
    recovered statically as `params.m.shape[0]` (see :func:`n_links`).

    Fields
    ------
    m         : (n,)      link masses $m_i$
    ell       : (n, 3)    joint $i\to$ joint $i{+}1$ offset $\ell_i$, link-$i$ frame.
                          `ell[-1]` is unused (no downstream link) — keep it zero.
    c         : (n, 3)    joint $i\to$ COM offset $c_i$, link-$i$ frame
    I_body    : (n, 3, 3) inertia $\mathbb{I}_i$ about the COM, link-$i$ frame
    d         : (n,)      joint viscous friction $d_i$ (acts on relative rate)
    kappa     : (n,)      air drag $\kappa_i$ (acts on absolute rate). Default 0.
    a         : (n,)      wind drag area $a_i$
    g         : ()        gravitational acceleration
    gain      : (n, 3)    per-axis actuator gain $\gamma_i$, so the input map is
                          $g(q)=T(q)^\top\mathrm{blkdiag}(\mathrm{diag}(\gamma_i))$.
                          Mirrors the single pendulum's `g_diag`; the default
                          $\gamma_i=(1,1,1)$ recovers the pure-kinematic
                          $g(q)=T(q)^\top$ exactly. **Not** to be confused with
                          `g` above (gravity) — see :func:`input_map`.
    varying_friction : () friction-modulation gate $\in\{0,1\}$ as a float, so the
                          model stays branch-free under `jit`. See
                          :func:`friction_modulation`.
    """
    m:      jnp.ndarray
    ell:    jnp.ndarray
    c:      jnp.ndarray
    I_body: jnp.ndarray
    d:      jnp.ndarray
    kappa:  jnp.ndarray
    a:      jnp.ndarray
    g:      jnp.ndarray
    gain:   jnp.ndarray
    varying_friction: jnp.ndarray


def n_links(params: ArmParams) -> int:
    """Number of links $n$ — a *static* Python int (safe for shapes under jit)."""
    return int(params.m.shape[0])


def uniform_chain_params(
    n: int,
    *,
    m: float = 1.0,
    link_length: float = 1.0,
    com_fraction: float = 1.0,
    inertia_scale: float = 1.0,
    d: float = 0.5,
    kappa: float = 0.0,
    a: float = 1.0,
    g: float = 9.81,
    g_diag=(1.0, 1.0, 1.0),
    varying_friction: bool = False,
    dtype=jnp.float64,
) -> ArmParams:
    r"""Build a chain of $n$ identical links along the body $z$-axis.

    Geometry: each link runs from its joint along $+e_z$, so
    $\ell_i = L\,e_z$ and $c_i = \alpha L\,e_z$ with $\alpha=$ `com_fraction`
    ($\alpha=1$ puts the mass at the tip — a bob; $\alpha=\tfrac12$ is a
    uniform rod).

    `inertia_scale` sets $\mathbb{I}_i = \mathrm{diag}(0,0,\,\text{scale}\cdot m L^2)$,
    i.e. **axial spin inertia** about the link's own $z$-axis.

    .. warning::
       `inertia_scale` must be $>0$ or $M(q)$ is **singular**. With all levers
       along $e_z$ ($\ell_i, c_i \parallel e_z$), both $[\ell_i]_\times$ and
       $[c_i]_\times$ annihilate $e_z$, so the translational term contributes
       nothing to each link's axial spin: a point mass on its own axis has no
       inertia about that axis. Ball joints leave that spin free, so without
       axial inertia $M$ has $n$ zero eigenvalues and $M^{-1}p$ blows up.
       (Revolute joints do not suffer this — they constrain the spin away.)

       The default `inertia_scale=1.0` with `com_fraction=1.0` gives
       $M = m L^2 I_3$ at $n=1$, reproducing the existing single pendulum
       exactly (see :func:`envs.arm_nlink_SO3.windy_arm_nlink_so3.pendulum_equivalent_params`
       and §13.1 of `multi-joint-ph-system.md`).
    """
    L = float(link_length)
    ez = jnp.array([0.0, 0.0, 1.0], dtype=dtype)

    ell = jnp.tile(L * ez, (n, 1))
    # The last link has no downstream neighbour, so its tip offset never enters
    # the dynamics. Zero it so a bug that *does* use it shows up loudly.
    ell = ell.at[-1].set(0.0)

    c = jnp.tile(com_fraction * L * ez, (n, 1))

    I_one = jnp.diag(jnp.array([0.0, 0.0, inertia_scale * m * L ** 2], dtype=dtype))
    I_body = jnp.tile(I_one, (n, 1, 1))

    # Actuator gain: a scalar broadcasts to every axis of every link, a 3-vector
    # to every link (the pendulum's `g_diag` convention), or pass an (n,3) array
    # for per-link gains.
    gain = jnp.broadcast_to(jnp.asarray(g_diag, dtype=dtype), (n, 3))

    ones = jnp.ones((n,), dtype=dtype)
    return ArmParams(
        m=m * ones,
        ell=ell.astype(dtype),
        c=c.astype(dtype),
        I_body=I_body.astype(dtype),
        d=d * ones,
        kappa=kappa * ones,
        a=a * ones,
        g=jnp.asarray(g, dtype=dtype),
        gain=gain.astype(dtype),
        varying_friction=jnp.asarray(1.0 if varying_friction else 0.0, dtype=dtype),
    )


# ══════════════════════════════════════════════════════════════════════
# Small array helpers
# ══════════════════════════════════════════════════════════════════════

def _exclusive_prefix_sum(x: jnp.ndarray) -> jnp.ndarray:
    r"""`out[i] = ` $\sum_{j<i} x_j$ along axis 0 (`out[0]` is zero)."""
    zero = jnp.zeros((1,) + x.shape[1:], dtype=x.dtype)
    return jnp.concatenate([zero, jnp.cumsum(x, axis=0)[:-1]], axis=0)


def _strict_suffix_sum(x: jnp.ndarray) -> jnp.ndarray:
    r"""`out[j] = ` $\sum_{i>j} x_i$ along axis 0 (`out[-1]` is zero)."""
    inclusive = jnp.cumsum(x[::-1], axis=0)[::-1]          # $\sum_{i\ge j}$
    zero = jnp.zeros((1,) + x.shape[1:], dtype=x.dtype)
    return jnp.concatenate([inclusive[1:], zero], axis=0)


def block_diag_3(blocks: jnp.ndarray) -> jnp.ndarray:
    r"""`(n, 3, 3)` blocks $\to$ the $(3n, 3n)$ block-diagonal matrix."""
    n = blocks.shape[0]
    out = jnp.zeros((n, 3, n, 3), dtype=blocks.dtype)
    idx = jnp.arange(n)
    out = out.at[idx, :, idx, :].set(blocks)
    return out.reshape(3 * n, 3 * n)


def blocks_to_matrix(blocks: jnp.ndarray) -> jnp.ndarray:
    r"""`(n, n, 3, 3)` blocks $\to$ the dense $(3n, 3n)$ matrix.

    `blocks[i, j]` becomes the $(i,j)$-th $3\times3$ block.
    """
    n = blocks.shape[0]
    return jnp.transpose(blocks, (0, 2, 1, 3)).reshape(3 * n, 3 * n)


# ══════════════════════════════════════════════════════════════════════
# 1. Kinematics  (§3 of the write-up)
# ══════════════════════════════════════════════════════════════════════

def joint_positions(params: ArmParams, R: jnp.ndarray) -> jnp.ndarray:
    r"""World positions of the joints, $o_i = \sum_{j<i} R_j \ell_j$, `(n, 3)`.

    Joint 1 is anchored at the origin, so `o[0] == 0`.
    """
    R_ell = jnp.einsum('jab,jb->ja', R, params.ell)          # $R_j\ell_j$
    return _exclusive_prefix_sum(R_ell)


def com_positions(params: ArmParams, R: jnp.ndarray) -> jnp.ndarray:
    r"""World COM positions, $p_i = \sum_{j<i}R_j\ell_j + R_i c_i$, `(n, 3)`."""
    return joint_positions(params, R) + jnp.einsum('iab,ib->ia', R, params.c)


def translational_jacobians(params: ArmParams, R: jnp.ndarray) -> jnp.ndarray:
    r"""Stack of $J_{v,i}(q)\in\mathbb{R}^{3\times3n}$ with $\dot p_i = J_{v,i}\,\omega$.

    From $\dot R_j = R_j[\omega_j]_\times$ and $a\times b = -[b]_\times a$:

    .. math::
        \dot p_i = -\sum_{j<i}R_j[\ell_j]_\times\omega_j - R_i[c_i]_\times\omega_i

    so $J_{v,i}$ is **block lower-triangular**:

    .. math::
        [J_{v,i}]_j = \begin{cases}
            -R_j[\ell_j]_\times & j<i \\
            -R_i[c_i]_\times    & j=i \\
            0                   & j>i
        \end{cases}

    Returns `(n, 3, 3n)`.
    """
    n = n_links(params)

    # $-R_j[\ell_j]_\times$ — the block used whenever $j<i$ (depends on $j$ only).
    off_blocks = -jnp.einsum('jab,jbc->jac', R, hat_batch(params.ell))   # (n,3,3)
    # $-R_i[c_i]_\times$ — the diagonal block (depends on $i$ only).
    dia_blocks = -jnp.einsum('iab,ibc->iac', R, hat_batch(params.c))     # (n,3,3)

    i_idx = jnp.arange(n)[:, None]                       # (n, 1)
    j_idx = jnp.arange(n)[None, :]                       # (1, n)
    is_lower = (j_idx < i_idx)[..., None, None]          # (n, n, 1, 1)
    is_diag = (j_idx == i_idx)[..., None, None]

    blocks = jnp.where(
        is_lower,
        off_blocks[None, :, :, :],                       # indexed by $j$
        jnp.where(is_diag, dia_blocks[:, None, :, :], 0.0),   # indexed by $i$
    )                                                    # (n, n, 3, 3)

    # (i, j, 3, 3) -> (i, 3, j, 3) -> (n, 3, 3n)
    return jnp.transpose(blocks, (0, 2, 1, 3)).reshape(n, 3, 3 * n)


def joint_rate_map(R: jnp.ndarray) -> jnp.ndarray:
    r"""$T(q)$ mapping absolute rates to relative (joint) rates, $\Omega = T(q)\omega$.

    .. math::
        \Omega_i = \omega_i - R_i^\top R_{i-1}\,\omega_{i-1},\qquad \omega_0 := 0

    so $T$ is **block lower-bidiagonal** with $T_{ii}=I_3$ and
    $T_{i,i-1} = -R_i^\top R_{i-1}$. Note $\det T = 1$, so $T$ is always
    invertible — this is why joint friction alone damps every direction.

    Returns `(3n, 3n)`.
    """
    n = R.shape[0]
    blocks = jnp.zeros((n, n, 3, 3), dtype=R.dtype)
    idx = jnp.arange(n)
    blocks = blocks.at[idx, idx].set(jnp.eye(3, dtype=R.dtype))

    if n > 1:
        # sub[k] = $R_{k+1}^\top R_k$  (0-indexed links)
        sub = -jnp.einsum('kab,kac->kbc', R[1:], R[:-1])          # (n-1, 3, 3)
        blocks = blocks.at[jnp.arange(1, n), jnp.arange(0, n - 1)].set(sub)

    return blocks_to_matrix(blocks)


def input_map(params: ArmParams, R: jnp.ndarray) -> jnp.ndarray:
    r"""$g(q) = T(q)^\top\,\Gamma$ — joint torques to generalized force. `(3n, 3n)`.

    With unit gain the map is pure kinematics: power at joint $i$ is
    $u_i^\top\Omega_i$, so $u^\top\Omega = u^\top T\omega = (T^\top u)^\top\omega$,
    giving $g = T^\top$ and

    .. math::
        \big(g(q)u\big)_i = u_i - R_i^\top R_{i+1}\,u_{i+1},\qquad u_{n+1}:=0 .

    The second term is **not** a second control — it is Newton's third law: the
    motor at joint $i{+}1$ reacts on link $i$. The system is fully actuated
    ($3n$ inputs for $3n$ DOF); drop columns for under-actuation.

    A per-axis **actuator gain** $\Gamma=\mathrm{blkdiag}(\mathrm{diag}(\gamma_i))$
    scales each commanded torque before it reaches the joint, so the delivered
    torque is $\Gamma u$ and the power identity becomes
    $(\Gamma u)^\top\Omega=(gu)^\top\omega$. This mirrors `g_diag` in the single
    pendulum (``envs/windy_pendulum_3d.py``): without it $g$ is fully determined
    by the state and there is nothing for a model to identify, which would make
    the arm a strictly easier problem than the pendulum. $\gamma_i=(1,1,1)$
    recovers $T(q)^\top$ exactly.

    .. note::
       $\Gamma$ multiplies on the **right**, so $g$'s columns scale (per input
       channel), not its rows. $D_{\mathrm{joint}}$ is built from $T$, never from
       $g$ — see :func:`dissipation_matrix`.
    """
    return joint_rate_map(R).T * params.gain.reshape(-1)[None, :]


# ══════════════════════════════════════════════════════════════════════
# 2. Energy  (§4 of the write-up)
# ══════════════════════════════════════════════════════════════════════

def mass_matrix(params: ArmParams, R: jnp.ndarray) -> jnp.ndarray:
    r"""Mass matrix $M(q)\in\mathbb{R}^{3n\times3n}$, SPD, with $T=\tfrac12\omega^\top M\omega$.

    .. math::
        M(q) = \mathrm{blkdiag}(\mathbb{I}_1,\dots,\mathbb{I}_n)
             + \sum_i m_i\,J_{v,i}(q)^\top J_{v,i}(q)

    In absolute coordinates the rotational part is a *constant* block diagonal —
    all the $q$-dependence sits in the translational Jacobians.

    Invariance: $M_{jk}$ depends on $q$ only through the relative rotations
    $R_j^\top R_k$, hence $M(hq) = M(q)$ for any global $h\in SO(3)$.
    """
    Jv = translational_jacobians(params, R)                       # (n, 3, 3n)
    translational = jnp.einsum('i,iaj,iak->jk', params.m, Jv, Jv)
    return block_diag_3(params.I_body) + translational


def potential(params: ArmParams, R: jnp.ndarray) -> jnp.ndarray:
    r"""Gravitational potential $V(q)$ (a scalar).

    .. math::
        V(q) = \sum_i m_i g\,\langle e_z, p_i(q)\rangle
             = g\,e_z^\top\sum_j R_j\Big(\mu_j^{>}\ell_j + m_j c_j\Big),
        \qquad \mu_j^{>} = \sum_{i>j} m_i

    Read: link $j$ carries its own COM plus all the mass hanging off its tip.
    """
    mu_downstream = _strict_suffix_sum(params.m)                   # (n,)
    lever = mu_downstream[:, None] * params.ell + params.m[:, None] * params.c
    heights = jnp.einsum('jab,jb->ja', R, lever)[:, 2]             # $e_z^\top R_j(\cdot)$
    return params.g * jnp.sum(heights)


def hamiltonian(params: ArmParams, R: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
    r"""$H(q,p) = \tfrac12 p^\top M^{-1}(q)p + V(q)$ (a scalar).

    $M^{-1}p$ is evaluated with a linear **solve** rather than by forming the
    explicit inverse — better conditioned, and `jnp.linalg.solve` is
    differentiable, which is what `drift_p` relies on.
    """
    p_flat = p.reshape(-1)
    xi = jnp.linalg.solve(mass_matrix(params, R), p_flat)
    return 0.5 * jnp.dot(p_flat, xi) + potential(params, R)


def momentum_from_omega(params: ArmParams, R: jnp.ndarray,
                        omega: jnp.ndarray) -> jnp.ndarray:
    r"""$p = M(q)\,\omega$. Shapes `(n, 3) -> (n, 3)`."""
    n = n_links(params)
    return (mass_matrix(params, R) @ omega.reshape(-1)).reshape(n, 3)


def omega_from_momentum(params: ArmParams, R: jnp.ndarray,
                        p: jnp.ndarray) -> jnp.ndarray:
    r"""$\omega = \xi = M^{-1}(q)\,p$. Shapes `(n, 3) -> (n, 3)`."""
    n = n_links(params)
    return jnp.linalg.solve(mass_matrix(params, R), p.reshape(-1)).reshape(n, 3)


# ══════════════════════════════════════════════════════════════════════
# 3. Dissipation  (§7 of the write-up)
# ══════════════════════════════════════════════════════════════════════

def friction_modulation(params: ArmParams, R: jnp.ndarray,
                        omega: jnp.ndarray) -> jnp.ndarray:
    r"""Per-link friction multiplier $\rho_i \ge 1$, `(n,)`.

    Generalizes the single-pendulum modulation in
    ``envs/windy_pendulum_3d.py::_variable_friction``:

    .. math::
        \rho_i = 1 + \tfrac12\underbrace{\tfrac12\big(1 - (R_ie_z)_z\big)}_{\text{height}}
                   + \tfrac12\tanh\lVert\omega_i\rVert

    The whole correction is gated by `params.varying_friction` $\in\{0,1\}$ as a
    float so the function stays branch-free (jit-safe); the gate at 0 gives
    $\rho_i \equiv 1$ exactly.
    """
    height = 0.5 * (1.0 - jnp.einsum('iab,b->ia', R, E_Z.astype(R.dtype))[:, 2])
    speed = jnp.tanh(jnp.linalg.norm(omega, axis=-1))
    return 1.0 + params.varying_friction * (0.5 * height + 0.5 * speed)


def dissipation_matrix(params: ArmParams, R: jnp.ndarray,
                       omega: jnp.ndarray) -> jnp.ndarray:
    r"""$D(q,\omega)\succeq 0$, `(3n, 3n)`, the Rayleigh dissipation operator.

    Two physically distinct contributions:

    * **Joint friction** — opposes the *relative* rate $\Omega = T(q)\omega$:

      .. math:: D_{\mathrm{joint}}(q) = T(q)^\top\,\mathrm{blkdiag}(\rho_i d_i I_3)\,T(q)

      Since $T$ is invertible, $D_{\mathrm{joint}}\succ0$ whenever all $d_i>0$ —
      joint friction alone damps every direction.

    * **Air drag** — opposes the *absolute* rate $\omega$:

      .. math:: D_{\mathrm{air}} = \mathrm{blkdiag}(\rho_i \kappa_i I_3)

      Defaults to zero (`kappa=0`); kept in the code path so it can be switched on.

    .. note::
       With **unit** actuator gain $D_{\mathrm{joint}} = g(q)\,\mathrm{blkdiag}
       (\rho_i d_i I_3)\,g(q)^\top$, i.e. joint friction is negative output
       feedback through the actuation port. That identity **fails** once
       $\Gamma\neq I$, because $g=T^\top\Gamma$ picks up the gain while friction
       does not — friction opposes the physical joint rate regardless of how
       hard the motor pushes. $D$ is therefore always built from $T$, never
       from :func:`input_map`.
    """
    rho = friction_modulation(params, R, omega)                    # (n,)
    eye = jnp.eye(3, dtype=R.dtype)

    T = joint_rate_map(R)
    D_joint_coeff = block_diag_3((rho * params.d)[:, None, None] * eye)
    D_joint = T.T @ D_joint_coeff @ T

    D_air = block_diag_3((rho * params.kappa)[:, None, None] * eye)
    return D_joint + D_air


# ══════════════════════════════════════════════════════════════════════
# 4. Wind  (§9 of the write-up)
# ══════════════════════════════════════════════════════════════════════

def wrench_to_joint_torque(params: ArmParams, R: jnp.ndarray,
                           f: jnp.ndarray) -> jnp.ndarray:
    r"""Map world-frame COM forces $\{f_i\}$ to generalized force, $\tau=\sum_i J_{v,i}^\top f_i$.

    .. math::
        \tau_j = [\ell_j]_\times R_j^\top\!\!\sum_{i>j}f_i \;+\; [c_j]_\times R_j^\top f_j

    Every body force in the system flows through this one map — gravity
    ($f_i = -m_i g e_z$) and wind ($f_i = a_i F$) alike. Shapes `(n,3) -> (n,3)`.
    """
    downstream = _strict_suffix_sum(f)                             # (n, 3)

    # $R_j^\top(\cdot)$ for the downstream resultant and for the own-COM force.
    body_down = jnp.einsum('jab,ja->jb', R, downstream)
    body_own = jnp.einsum('jab,ja->jb', R, f)

    return (jnp.cross(params.ell, body_down, axis=-1)
            + jnp.cross(params.c, body_own, axis=-1))


def wind_map(params: ArmParams, R: jnp.ndarray) -> jnp.ndarray:
    r"""$\Sigma(q)\in\mathbb{R}^{3n\times3}$ — lever map of a single world wind field.

    Substituting $f_i = a_i F$ into :func:`wrench_to_joint_torque`:

    .. math::
        \Sigma(q)_j = \Big(A_j^{>}[\ell_j]_\times + a_j[c_j]_\times\Big)R_j^\top,
        \qquad A_j^{>} = \sum_{i>j}a_i

    **Low rank by construction**: only 3 Brownian motions drive $3n$ momentum
    dimensions, because physically there is one wind field, not $n$ of them.
    """
    A_downstream = _strict_suffix_sum(params.a)                    # (n,)
    lever = (A_downstream[:, None, None] * hat_batch(params.ell)
             + params.a[:, None, None] * hat_batch(params.c))      # (n, 3, 3)
    blocks = jnp.einsum('jab,jcb->jac', lever, R)                  # $(\cdot)R_j^\top$
    return blocks.reshape(3 * R.shape[0], 3)


def analytic_gravity_torque(params: ArmParams, R: jnp.ndarray) -> jnp.ndarray:
    r"""Closed-form $\mathcal T_j(V)$ — the gravity generalized force, `(n, 3)`.

    .. math::
        \mathcal T_j(V) = -g\Big(\mu_j^{>}[\ell_j]_\times + m_j[c_j]_\times\Big)R_j^\top e_z

    Used **only** as an independent reference in the test suite; the dynamics
    obtain the same quantity by autodiff through $V$.
    """
    forces = -params.m[:, None] * params.g * E_Z.astype(R.dtype)[None, :]
    return wrench_to_joint_torque(params, R, forces)


# ══════════════════════════════════════════════════════════════════════
# 5. Dynamics  (§5, §6 of the write-up)
# ══════════════════════════════════════════════════════════════════════

def trivialized_grad(R: jnp.ndarray, dFdR: jnp.ndarray) -> jnp.ndarray:
    r"""The operator $\mathcal T_i(F) = \sum_k r_{ik}\times \partial F/\partial r_{ik}$.

    With $r_{ik}^\top$ the $k$-th **row** of $R_i$, perturbing
    $R_i \mapsto R_i\exp([\phi_i]_\times)$ moves row $k$ by $r_{ik}\times\phi_i$, so

    .. math:: \mathcal T_i(F) = -\,\partial F/\partial\phi_i .

    This is a directional derivative *along the group action*, hence independent
    of how $F$ is extended off $SO(3)^n$ — which is what makes the redundant
    $\mathbb{R}^{9n}$ embedding safe.

    Shapes: `R (n,3,3)`, `dFdR (n,3,3)` -> `(n, 3)`.
    """
    return jnp.sum(jnp.cross(R, dFdR, axis=-1), axis=1)


def dH_dq(params: ArmParams, R: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
    r"""$\partial H/\partial q$ in the flat embedding, `(n, 3, 3)`, with $p$ held fixed."""
    n = n_links(params)

    def H_of_q(q_flat):
        return hamiltonian(params, q_flat.reshape(n, 3, 3), p)

    return jax.grad(H_of_q)(R.reshape(-1)).reshape(n, 3, 3)


def drift_p(params: ArmParams, R: jnp.ndarray, p: jnp.ndarray,
            u: jnp.ndarray, wind_force: jnp.ndarray) -> jnp.ndarray:
    r"""Deterministic momentum drift $\dot p$, `(n, 3)`.

    .. math::
        \dot p_i = \underbrace{p_i\times\xi_i}_{\mathrm{ad}^*_\xi p}
                 + \underbrace{\mathcal T_i(H)}_{\text{gravity + inter-link Coriolis}}
                 - \underbrace{(D\xi)_i}_{\text{dissipation}}
                 + \underbrace{(g(q)u)_i}_{\text{actuation}}
                 + \underbrace{(\Sigma(q)F_{\mathrm{det}})_i}_{\text{deterministic wind}}

    You never write Christoffel symbols: the full Coriolis field is split between
    the gyroscopic term and $\mathcal T_i\big(\tfrac12 p^\top M^{-1}p\big)$, the
    latter produced by one autodiff pass through $M^{-1}(q)$.

    Args:
        u          : `(n, 3)` joint torques (link-$i$ body frame)
        wind_force : `(3,)` deterministic world-frame wind force $F_{\rm det}$
    """
    n = n_links(params)

    xi = omega_from_momentum(params, R, p)                         # $\xi = M^{-1}p$
    gyroscopic = jnp.cross(p, xi, axis=-1)                         # $p_i\times\xi_i$
    conservative = trivialized_grad(R, dH_dq(params, R, p))        # $\mathcal T_i(H)$

    D = dissipation_matrix(params, R, xi)
    damping = (D @ xi.reshape(-1)).reshape(n, 3)

    actuation = (input_map(params, R) @ u.reshape(-1)).reshape(n, 3)
    wind = (wind_map(params, R) @ wind_force).reshape(n, 3)

    return gyroscopic + conservative - damping + actuation + wind


def stochastic_increment_p(params: ArmParams, R: jnp.ndarray,
                           sigma: jnp.ndarray, dW: jnp.ndarray) -> jnp.ndarray:
    r"""Momentum increment from the wind noise, $\sigma\,\Sigma(q)\,dW$, `(n, 3)`.

    `dW` is a `(3,)` Wiener increment **already scaled** by $\sqrt{h}$ so that
    $\mathrm{Var}(dW) = h$ — matching the convention in
    ``envs/windy_pendulum_3d.py`` and ``src/utils/JAX/lie_integrator.py``.
    """
    n = n_links(params)
    return (sigma * (wind_map(params, R) @ dW)).reshape(n, 3)


# ══════════════════════════════════════════════════════════════════════
# 6. Geometric integrator  (§14 of the write-up)
# ══════════════════════════════════════════════════════════════════════

def lie_heun_step(params: ArmParams, R: jnp.ndarray, p: jnp.ndarray,
                  u: jnp.ndarray, h: jnp.ndarray, wind_force: jnp.ndarray,
                  sigma: jnp.ndarray, dW: jnp.ndarray):
    r"""One Stratonovich Lie–Heun substep on $SO(3)^n\times\mathbb{R}^{3n}$.

    .. math::
        \textbf{(1)}\;\; & \xi^{(1)}=M^{-1}(q)p,\quad
            \dot p^{(1)},\quad \Delta p^{(1)}=\sigma\Sigma(q)dW,\quad
            \phi^{(1)}_i=\xi^{(1)}_i h \\
        \textbf{(2)}\;\; & R_i^{\rm pr}=R_i\exp([\phi^{(1)}_i]_\times),\qquad
            p^{\rm pr}=p+\dot p^{(1)}h+\Delta p^{(1)} \\
        \textbf{(3)}\;\; & \text{re-evaluate at } (q^{\rm pr},p^{\rm pr})
            \text{ with the \emph{same} } dW \\
        \textbf{(4)}\;\; & \bar\phi_i=\tfrac12(\phi^{(1)}_i+\phi^{(2)}_i),\quad
            R_i^{+}=R_i\exp([\bar\phi_i]_\times),\\
        & p^{+}=p+\tfrac h2(\dot p^{(1)}+\dot p^{(2)})
                 +\tfrac12(\Delta p^{(1)}+\Delta p^{(2)})

    Every $R_i$ stays on $SO(3)$ **by construction** — one exponential per link
    per step, averaging done in the Lie *algebra*, never on matrices. No SVD
    re-projection is needed anywhere in the loop.

    Reusing the same `dW` across both stages is what makes this the Stratonovich
    (not Itô) scheme, matching ``envs/windy_pendulum_3d.py::_lie_heun_step``.
    """
    # ── Stage 1: evaluate at the current state ──
    p_dot_1 = drift_p(params, R, p, u, wind_force)
    xi_1 = omega_from_momentum(params, R, p)
    dp_stoch_1 = stochastic_increment_p(params, R, sigma, dW)
    phi_1 = xi_1 * h

    # ── Stage 2: predictor (Euler on the manifold) ──
    R_pred = R @ exp_so3_batch(phi_1)
    p_pred = p + p_dot_1 * h + dp_stoch_1

    # ── Stage 3: re-evaluate at the predicted state (same dW) ──
    p_dot_2 = drift_p(params, R_pred, p_pred, u, wind_force)
    xi_2 = omega_from_momentum(params, R_pred, p_pred)
    dp_stoch_2 = stochastic_increment_p(params, R_pred, sigma, dW)
    phi_2 = xi_2 * h

    # ── Stage 4: corrector (average in the algebra, exponentiate once) ──
    phi_avg = 0.5 * (phi_1 + phi_2)
    R_new = R @ exp_so3_batch(phi_avg)
    p_new = (p
             + 0.5 * (p_dot_1 + p_dot_2) * h
             + 0.5 * (dp_stoch_1 + dp_stoch_2))
    return R_new, p_new


def lie_heun_substeps(params: ArmParams, R: jnp.ndarray, p: jnp.ndarray,
                      u: jnp.ndarray, h: jnp.ndarray, wind_force: jnp.ndarray,
                      sigma: jnp.ndarray, dW_seq: jnp.ndarray):
    r"""Scan :func:`lie_heun_step` over `dW_seq` of shape `(n_substeps, 3)`.

    The control `u` and deterministic wind are held constant across the substeps,
    matching the ``step()`` semantics of the existing environment.
    """
    def body(carry, dW):
        R_c, p_c = carry
        return lie_heun_step(params, R_c, p_c, u, h, wind_force, sigma, dW), None

    (R_out, p_out), _ = jax.lax.scan(body, (R, p), dW_seq)
    return R_out, p_out


@jax.jit
def step_qomega(params: ArmParams, R: jnp.ndarray, omega: jnp.ndarray,
                u: jnp.ndarray, h: jnp.ndarray, wind_force: jnp.ndarray,
                sigma: jnp.ndarray, dW_seq: jnp.ndarray):
    r"""One environment step in $(q,\omega)$ coordinates — the jitted entry point.

    Internally the scan carries the momentum $p$ (the natural pH state); the
    conversions $p = M(q)\omega$ and $\omega = M^{-1}(q)p$ happen only at the
    step boundaries, mirroring ``src/utils/JAX/lie_integrator.py``.
    """
    p0 = momentum_from_omega(params, R, omega)
    R_new, p_new = lie_heun_substeps(params, R, p0, u, h, wind_force, sigma, dW_seq)
    return R_new, omega_from_momentum(params, R_new, p_new)


# ══════════════════════════════════════════════════════════════════════
# 7. Diagnostics / conserved quantities  (§11, §20 of the write-up)
# ══════════════════════════════════════════════════════════════════════

def total_energy(params: ArmParams, R: jnp.ndarray,
                 omega: jnp.ndarray) -> jnp.ndarray:
    r"""$H = \tfrac12\omega^\top M(q)\omega + V(q)$ from $(q,\omega)$ (a scalar)."""
    omega_flat = omega.reshape(-1)
    kinetic = 0.5 * jnp.dot(omega_flat, mass_matrix(params, R) @ omega_flat)
    return kinetic + potential(params, R)


def vertical_angular_momentum(params: ArmParams, R: jnp.ndarray,
                              omega: jnp.ndarray) -> jnp.ndarray:
    r"""Noether charge of the residual $SO(2)$ symmetry about $e_z$ (a scalar).

    $V$ is invariant under a global rotation $R_i\mapsto R_z(\alpha)R_i$, whose
    trivialized generator is $\delta\phi_i = \alpha R_i^\top e_z$. The conserved
    momentum is therefore

    .. math:: J_z = e_z^\top \sum_i R_i\,p_i .

    Conserved exactly when $D=0$, $u=0$ and the wind is off. Independent of the
    energy test — it checks the *gravity* term specifically.
    """
    p = momentum_from_omega(params, R, omega)
    return jnp.sum(jnp.einsum('iab,ib->ia', R, p)[:, 2])


def so3_defect(R: jnp.ndarray):
    r"""Manifold-constraint residuals: $\max_i\lVert R_i^\top R_i-I\rVert_F$ and
    $\max_i|\det R_i-1|$. Both should stay at machine precision.
    """
    gram = jnp.einsum('iab,iac->ibc', R, R) - jnp.eye(3, dtype=R.dtype)
    orth = jnp.max(jnp.linalg.norm(gram.reshape(R.shape[0], -1), axis=-1))
    det = jnp.max(jnp.abs(jnp.linalg.det(R) - 1.0))
    return orth, det


# ══════════════════════════════════════════════════════════════════════
# Self-check
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    jax.config.update('jax_enable_x64', True)

    n = 3
    params = uniform_chain_params(n, d=0.3, varying_friction=True)
    key = jax.random.PRNGKey(0)
    kR, kw = jax.random.split(key)

    # Random attitudes via the exponential map (guaranteed on SO(3)).
    R = exp_so3_batch(jax.random.normal(kR, (n, 3)))
    omega = jax.random.normal(kw, (n, 3)) * 0.5
    u = jnp.zeros((n, 3))

    print(f'n = {n}')
    print(f'  M(q)          {mass_matrix(params, R).shape}   '
          f'sym err = {jnp.max(jnp.abs(mass_matrix(params, R) - mass_matrix(params, R).T)):.2e}   '
          f'min eig = {jnp.min(jnp.linalg.eigvalsh(mass_matrix(params, R))):.4f}')
    print(f'  D(q,w)        {dissipation_matrix(params, R, omega).shape}   '
          f'min eig = {jnp.min(jnp.linalg.eigvalsh(dissipation_matrix(params, R, omega))):.4f}')
    print(f'  g(q)          {input_map(params, R).shape}')
    print(f'  Sigma(q)      {wind_map(params, R).shape}   '
          f'rank = {jnp.linalg.matrix_rank(wind_map(params, R))}')
    print(f'  V(q)          {potential(params, R):.6f}')
    print(f'  H(q,p)        {total_energy(params, R, omega):.6f}')
    print(f'  J_z           {vertical_angular_momentum(params, R, omega):.6f}')
    print(f'  drift_p       {drift_p(params, R, momentum_from_omega(params, R, omega), u, jnp.zeros(3)).shape}')
