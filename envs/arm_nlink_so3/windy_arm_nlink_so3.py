r"""Gymnasium environment: $n$-link windy robotic arm on $SO(3)^n$.

A thin, stateful shell around :mod:`envs.arm_nlink_so3.arm_nlink_physics`, which holds all the
mathematics. This file only manages state, randomness, the Gym API and rendering.

Generalizes ``envs/pendulum_so3/windy_pendulum_3d.py`` from one rigid body to a serial chain
of $n$ **ball-jointed** links. At $n=1$ it reproduces that environment exactly —
see :func:`pendulum_equivalent_params` and the regression test in
``mini_tests/test_gt_pH_matches_arm_env.py``.

State and observation
---------------------
    R      : (n, 3, 3)   absolute (world) attitude of each link
    omega  : (n, 3)      body-frame angular rate of each link
    obs    : (12n,)      concat(R.reshape(-1), omega.reshape(-1))

Action: `u` of shape `(n, 3)` or `(3n,)` — the actuator torque at each joint,
expressed in that link's body frame. The arm is **fully actuated**: $3n$ inputs
for $3n$ DOF.

Wind (§9 of `multi-joint-ph-system.md`)
---------------------------------------
A single world-frame wind field acts on every link and reaches the momentum
through the shared lever map $\Sigma(q)\in\mathbb{R}^{3n\times3}$:

.. math::
    F(t)\,dt = \underbrace{w(t)\,\mathrm{d}\,dt}_{\texttt{external\_force\_std}}
             + \underbrace{\sigma\,dW_t}_{\texttt{wind\_force\_std}}

Only **3 Brownian motions** drive all $3n$ momentum dimensions, because
physically there is one wind field, not $n$ independent ones.

Defaults chosen so the deterministic/dissipative wind-like channels are present
in the code path but contribute nothing until switched on:

    kappa (air drag)      = 0.0
    external_force_std    = 0.0     (deterministic wind off)
    wind_force_std        = 0.0     (set > 0 for the stochastic wind)
"""
from __future__ import annotations

import os
import sys
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_FILE_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.arm_nlink_so3.arm_nlink_physics import (            # noqa: E402
    ArmParams, uniform_chain_params, n_links,
    step_qomega, com_positions, joint_positions,
    total_energy, vertical_angular_momentum, so3_defect,
)


# ══════════════════════════════════════════════════════════════════════
# Parameter helpers
# ══════════════════════════════════════════════════════════════════════

def pendulum_equivalent_params(m: float = 1.0, l: float = 1.0,
                               g: float = 9.81, friction_coeff: float = 0.5,
                               varying_friction: bool = False,
                               g_diag=(1.0, 1.0, 1.0)) -> ArmParams:
    r"""The $n=1$ parameters that reproduce ``windy_pendulum_3d`` **exactly**.

    Per §13.1 of `multi-joint-ph-system.md`, setting $c_1 = \ell e_z$ and
    $\mathbb{I}_1 = \mathrm{diag}(0,0,m\ell^2)$ gives

    .. math::
        M = \mathrm{diag}(0,0,m\ell^2) + m\ell^2\,\mathrm{diag}(1,1,0) = m\ell^2 I_3

    matching that environment's isotropic ``self.I``, together with
    $V = mg\ell R_{22}$, $\Sigma = [c_1]_\times R^\top$ and $D = d_1 I_3$
    (at $n=1$ there is no parent link, so $T=I_3$ and the joint friction acts
    directly on the absolute rate).

    The input map reduces to $g(q)=T(q)^\top\Gamma=\mathrm{diag}(\gamma)$, which
    is exactly how ``windy_pendulum_3d`` applies its own `g_diag`
    (``self.g_diag_mat @ u``), so the two agree for any gain — pass the
    pendulum's `g_diag` straight through.
    """
    return uniform_chain_params(
        1, m=m, link_length=l, com_fraction=1.0, inertia_scale=1.0,
        d=friction_coeff, kappa=0.0, a=1.0, g=g, g_diag=g_diag,
        varying_friction=varying_friction,
    )


def _random_rotations(rng: np.random.Generator, n: int) -> np.ndarray:
    """`n` uniform random rotation matrices (Shoemake's quaternion method)."""
    u = rng.random((n, 3))
    q1 = np.sqrt(1 - u[:, 0]) * np.sin(2 * np.pi * u[:, 1])
    q2 = np.sqrt(1 - u[:, 0]) * np.cos(2 * np.pi * u[:, 1])
    q3 = np.sqrt(u[:, 0]) * np.sin(2 * np.pi * u[:, 2])
    q4 = np.sqrt(u[:, 0]) * np.cos(2 * np.pi * u[:, 2])
    x, y, z, w = q1, q2, q3, q4
    R = np.empty((n, 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _project_to_so3(R: np.ndarray) -> np.ndarray:
    """Project `(n,3,3)` to the nearest rotations (polar decomposition).

    Only used to sanitise user-supplied initial conditions — the Lie-group
    integrator keeps `R` on the manifold by construction, so this is never
    needed inside the rollout.
    """
    U, _, Vt = np.linalg.svd(R)
    out = U @ Vt
    bad = np.linalg.det(out) < 0
    if np.any(bad):
        U[bad, :, -1] *= -1.0
        out = U @ Vt
    return out


# ══════════════════════════════════════════════════════════════════════
# Environment
# ══════════════════════════════════════════════════════════════════════

class windy_arm_nlink_so3(gym.Env):
    r"""$n$-link ball-jointed arm on $SO(3)^n$ with wind, friction and noise.

    The dynamics are the stochastic port-Hamiltonian system

    .. math::
        dR_i &= R_i[\xi_i]_\times dt,\qquad \xi = M^{-1}(q)p \\
        dp   &= \big(\hat P\xi + \mathcal T(H) - D\xi + g(q)u
                  + \Sigma(q)w(t)\mathrm d\big)dt + \sigma\Sigma(q)\circ dW_t

    integrated by the Stratonovich Lie–Heun scheme of §14, so every $R_i$ stays
    on $SO(3)$ to machine precision without re-projection.
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(
        self,
        n: int = 2,
        *,
        render_mode: Optional[str] = None,
        # ── geometry / inertia ──
        m: float = 1.0,
        link_length: float = 1.0,
        com_fraction: float = 1.0,
        inertia_scale: float = 1.0,
        g: float = 9.81,
        # Per-axis actuator gain, mirroring `g_diag` in windy_pendulum_3d. The
        # default (1,1,1) leaves g(q) = T(q)^T, i.e. fully determined by the
        # state; set it away from 1 to give a model something to identify.
        g_diag: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        # ── dissipation ──
        friction_coeff: float = 0.5,      # joint friction $d_i$
        air_drag: float = 0.0,            # $\kappa_i$ — present but off by default
        varying_friction: bool = False,
        # ── wind ──
        wind_drag_area: float = 1.0,      # $a_i$
        external_force_type: str = 'sine',
        external_force_std: float = 0.0,  # deterministic wind — off by default
        external_force_direction: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        wind_force_std: float = 0.0,      # $\sigma$ — the stochastic wind
        # ── integration ──
        dt: float = 0.05,
        n_substeps: int = 10,
        # ── misc ──
        max_speed: float = 8.0,
        max_torque: float = 2.0,
        params: Optional[ArmParams] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.n = int(n)
        self.render_mode = render_mode
        self.dt = float(dt)
        self.n_substeps = int(n_substeps)
        self.h = self.dt / self.n_substeps

        # `max_speed` / `max_torque` define the Gym API bounds and are used as a
        # soft validity filter by the dataset generator. Following
        # windy_pendulum_3d, neither is enforced by clipping in `step()`.
        self.max_speed = float(max_speed)
        self.max_torque = float(max_torque)

        self.params = params if params is not None else uniform_chain_params(
            self.n, m=m, link_length=link_length, com_fraction=com_fraction,
            inertia_scale=inertia_scale, d=friction_coeff, kappa=air_drag,
            a=wind_drag_area, g=g, g_diag=g_diag,
            varying_friction=varying_friction,
        )
        if n_links(self.params) != self.n:
            raise ValueError(
                f'params has {n_links(self.params)} links but n={self.n}')

        # Cheap guard against the singular-M failure mode documented in
        # `uniform_chain_params`: it is far easier to diagnose here than as a
        # NaN twenty integration steps later.
        self.link_length = float(link_length)

        self.external_force_type = str(external_force_type)
        self.external_force_std = float(external_force_std)
        d_vec = np.asarray(external_force_direction, dtype=np.float64)
        if np.linalg.norm(d_vec) < 1e-12:
            raise ValueError('external_force_direction must be non-zero.')
        self.external_force_direction = d_vec / np.linalg.norm(d_vec)
        self.wind_force_std = float(wind_force_std)

        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque,
            shape=(3 * self.n,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12 * self.n,), dtype=np.float32)

        self._np_rng = np.random.default_rng(seed)

        self.t = 0.0
        self.R = np.tile(np.eye(3), (self.n, 1, 1))
        self.omega = np.zeros((self.n, 3), dtype=np.float64)
        self.last_u = np.zeros((self.n, 3), dtype=np.float64)
        self.last_w = 0.0

        self._fig = None
        self._ax = None

    # ── Seeding & state access ────────────────────────────────────────

    def seed(self, seed: Optional[int] = None):
        self._np_rng = np.random.default_rng(seed)

    def get_state(self):
        """`(R, omega)` copies, shapes `(n,3,3)` and `(n,3)`."""
        return self.R.copy(), self.omega.copy()

    # ── Wind model ────────────────────────────────────────────────────

    def update_wind(self, t: float) -> float:
        r"""Deterministic wind amplitude $w(t)$ (scalar, times the fixed direction).

        Off by default (`external_force_std=0`). Matches the waveform menu of
        ``windy_pendulum_3d`` so datasets stay comparable.
        """
        s = self.external_force_std
        if self.external_force_type == 'sine':
            return s * np.sin(2 * np.pi * 0.5 * t)
        if self.external_force_type == 'square':
            return s * (1.0 if np.sin(2 * np.pi * 0.5 * t) >= 0 else -1.0)
        if self.external_force_type == 'random':
            return float(self._np_rng.normal(0.0, s)) if s > 0 else 0.0
        if self.external_force_type == 'constant':
            return s
        raise ValueError(f'Unknown external_force_type: {self.external_force_type}')

    # ── Observation ───────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        r"""$(\mathrm{vec}(R_1),\dots,\mathrm{vec}(R_n),\omega_1,\dots,\omega_n)$, `(12n,)`."""
        return np.concatenate([self.R.reshape(-1),
                               self.omega.reshape(-1)]).astype(np.float32)

    # ── Step ──────────────────────────────────────────────────────────

    def step(self, u):
        r"""Advance by `dt` using `n_substeps` Stratonovich Lie–Heun substeps.

        A fresh Wiener increment $dW\sim\mathcal N(0, h I_3)$ is drawn per
        substep — **3-dimensional**, because the wind is a single shared field
        (§9.3). Torque is held constant across the substeps.
        """
        u = np.asarray(u, dtype=np.float64).reshape(self.n, 3)
        self.last_u = u.copy()

        self.t += self.dt
        w = self.update_wind(self.t)
        self.last_w = float(w)
        wind_force = w * self.external_force_direction        # $F_{\rm det}\in\mathbb{R}^3$

        sigma = self.wind_force_std
        if sigma > 0.0:
            dW = self._np_rng.normal(0.0, np.sqrt(self.h), size=(self.n_substeps, 3))
        else:
            dW = np.zeros((self.n_substeps, 3))

        R_new, omega_new = step_qomega(
            self.params,
            jnp.asarray(self.R), jnp.asarray(self.omega),
            jnp.asarray(u), jnp.asarray(self.h),
            jnp.asarray(wind_force), jnp.asarray(sigma), jnp.asarray(dW),
        )
        self.R = np.asarray(R_new, dtype=np.float64)
        self.omega = np.asarray(omega_new, dtype=np.float64)

        # ── Reward: upright chain, small rates, small effort ──
        # $\text{cost} = \sum_i (1 - (R_ie_z)_z) + 0.1\lVert\omega\rVert^2 + 0.001\lVert u\rVert^2$
        angle_cost = float(np.sum(1.0 - self.R[:, 2, 2]))
        vel_cost = 0.1 * float(np.sum(self.omega ** 2))
        act_cost = 0.001 * float(np.sum(u ** 2))
        reward = -(angle_cost + vel_cost + act_cost)

        info = {'wind': self.last_w}
        return self._get_obs(), reward, False, False, info

    # ── Reset ─────────────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        self.t = 0.0
        self.last_u = np.zeros((self.n, 3), dtype=np.float64)
        self.last_w = 0.0
        options = options or {}

        if 'R_init' in options:
            R0 = np.asarray(options['R_init'], dtype=np.float64).reshape(self.n, 3, 3)
            R0 = _project_to_so3(R0)
        else:
            R0 = _random_rotations(self._np_rng, self.n)

        if 'omega_init' in options:
            w0 = np.asarray(options['omega_init'], dtype=np.float64).reshape(self.n, 3)
        else:
            w0 = self._np_rng.uniform(-1.0, 1.0, size=(self.n, 3))

        self.R = R0
        self.omega = w0
        return self._get_obs(), {}

    # ── Diagnostics ───────────────────────────────────────────────────

    def energy(self) -> float:
        r"""$H = \tfrac12\omega^\top M(q)\omega + V(q)$ at the current state."""
        return float(total_energy(self.params, jnp.asarray(self.R),
                                  jnp.asarray(self.omega)))

    def vertical_momentum(self) -> float:
        r"""$J_z = e_z^\top\sum_i R_i p_i$ — conserved when $D=u=\text{wind}=0$."""
        return float(vertical_angular_momentum(self.params, jnp.asarray(self.R),
                                               jnp.asarray(self.omega)))

    def manifold_defect(self) -> Tuple[float, float]:
        r"""$(\max_i\lVert R_i^\top R_i - I\rVert_F,\ \max_i|\det R_i - 1|)$."""
        orth, det = so3_defect(jnp.asarray(self.R))
        return float(orth), float(det)

    # ── Rendering ─────────────────────────────────────────────────────

    def _init_render(self):
        import matplotlib
        if self.render_mode == 'rgb_array':
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        self._fig = plt.figure(figsize=(7, 7))
        self._ax = self._fig.add_subplot(111, projection='3d')

    def render(self):
        if self.render_mode is None:
            return

        import matplotlib.pyplot as plt

        if self._fig is None or not plt.fignum_exists(self._fig.number):
            self._init_render()

        ax = self._ax
        ax.cla()

        R_j = jnp.asarray(self.R)
        joints = np.asarray(joint_positions(self.params, R_j))       # (n, 3)
        coms = np.asarray(com_positions(self.params, R_j))           # (n, 3)
        # Tip of the last link, so the chain is drawn all the way out.
        tip = coms[-1]
        chain = np.vstack([joints, tip])                             # (n+1, 3)

        ax.plot(chain[:, 0], chain[:, 1], chain[:, 2],
                color='#555555', linewidth=3, solid_capstyle='round', zorder=3)
        ax.scatter(coms[:, 0], coms[:, 1], coms[:, 2],
                   color='#cc4d4d', s=90, depthshade=True, zorder=5)
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
                   color='black', s=40, depthshade=True, zorder=5)

        reach = self.link_length * self.n
        if abs(self.last_w) > 1e-6:
            wv = self.last_w * self.external_force_direction * 0.5 * self.link_length
            ax.quiver(0, 0, 0, wv[0], wv[1], wv[2], color='#00bcd4',
                      linewidth=2.5, arrow_length_ratio=0.18,
                      label=f'wind = {self.last_w:+.2f}')
        ax.quiver(0, 0, 0, 0, 0, -0.4 * self.link_length, color='#999999',
                  linewidth=1.5, arrow_length_ratio=0.15, linestyle='dashed',
                  label='gravity')

        lim = 1.15 * reach
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_xlabel('X', fontsize=9)
        ax.set_ylabel('Y', fontsize=9)
        ax.set_zlabel('Z', fontsize=9)
        ax.set_title(f'{self.n}-link arm on SO(3)^{self.n} — Lie group integrator',
                     fontsize=12, fontweight='bold')
        ax.set_box_aspect([1, 1, 1])

        hud = (f't = {self.t:.2f}s    H = {self.energy():+.3f}    '
               f'|omega| = {np.linalg.norm(self.omega):.2f}')
        ax.text2D(0.02, 0.96, hud, transform=ax.transAxes, fontsize=9,
                  fontfamily='monospace', verticalalignment='top',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax.legend(loc='upper right', fontsize=8, framealpha=0.7)

        if self.render_mode == 'human':
            plt.draw()
            plt.pause(0.001)
            return None
        self._fig.canvas.draw()
        return np.asarray(self._fig.canvas.buffer_rgba())[:, :, :3].copy()

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
        self._fig = None
        self._ax = None


# ══════════════════════════════════════════════════════════════════════
# Demo
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    N = 2
    env = windy_arm_nlink_so3(
        n=N, dt=0.05, friction_coeff=0.0, air_drag=0.0,
        wind_force_std=0.0, external_force_std=0.0, seed=0,
    )
    env.reset(seed=0)

    print(f'{N}-link arm, no dissipation / no wind / no control')
    print(f'  obs dim = {env.observation_space.shape[0]}   '
          f'act dim = {env.action_space.shape[0]}')
    H0 = env.energy()
    Jz0 = env.vertical_momentum()
    print(f'  {"step":>5}  {"H":>12}  {"dH/H0":>10}  {"Jz":>12}  '
          f'{"dJz":>10}  {"orth":>9}  {"det":>9}')
    for k in range(201):
        if k % 40 == 0:
            orth, det = env.manifold_defect()
            print(f'  {k:>5}  {env.energy():>12.6f}  '
                  f'{abs(env.energy() - H0) / abs(H0):>10.2e}  '
                  f'{env.vertical_momentum():>12.6f}  '
                  f'{abs(env.vertical_momentum() - Jz0):>10.2e}  '
                  f'{orth:>9.2e}  {det:>9.2e}')
        env.step(np.zeros((N, 3)))
    env.close()
