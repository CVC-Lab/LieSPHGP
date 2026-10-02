r"""MuJoCo ball-joint arm wrapped in the *same* state convention as the analytic
env, so the learner cannot tell where its data came from.

The state convention
--------------------
Everything downstream — the dataset, ``utils/lie_integrator_nlink.py``, the
comparison report — speaks $(q,\omega)$ with

.. math::
    q = \big(\mathrm{vec}(R_1),\dots,\mathrm{vec}(R_n)\big)\in\mathbb{R}^{9n},
    \qquad \omega\in\mathbb{R}^{3n}

where $R_i$ is the **absolute** (world $\leftarrow$ body) rotation of link $i$
and $\omega_i$ is its **absolute angular velocity expressed in its own body
frame**. Note the model never sees momentum: the rollout converts internally
with $p = M_\theta(q)\,\omega$ (``lie_integrator_nlink._to_p``). That matters
here — it means building this dataset never requires the true mass matrix, so
no ground-truth parameter leaks into the training targets.

MuJoCo speaks something different
---------------------------------
MuJoCo's generalized coordinates for a ball-joint chain are **relative**:

* ``qpos`` per joint is the quaternion of $R_{i-1}^\top R_i$ (child w.r.t. parent)
* ``qvel`` per joint is the *relative* rate $\Omega_i$ in the child frame

and those are related to ours by exactly the joint-rate map already in the
codebase, $\Omega = T(q)\,\omega$ with $T_{ii}=I_3$,
$T_{i,i-1}=-R_i^\top R_{i-1}$ (:func:`arm_nlink_physics.joint_rate_map`):

.. math::
    \Omega_i \;=\; \omega_i - R_i^\top R_{i-1}\,\omega_{i-1},
    \qquad \omega_0 := 0 .

Since $\det T = 1$ this is always invertible, and the inverse is the forward
recursion $\omega_i = \Omega_i + R_i^\top R_{i-1}\omega_{i-1}$.

Trust, but verify
-----------------
"``qvel`` is the relative rate in the child frame" is an *assumption* about
MuJoCo's conventions, and getting it wrong would silently corrupt every dataset.
So :meth:`MujocoArmEnv.body_rates` computes $\omega$ **two independent ways** —
by the recursion above from ``qvel``, and from ``mj_objectVelocity`` with
``flg_local=1`` — and ``test_mujoco_matches_gt.py`` asserts they agree. If the
frame convention were different the two would diverge immediately.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mujoco                                                    # noqa: E402

from build_mjcf import ArmSpec, compile_spec, params_from_mjmodel, _body_id  # noqa: E402


class MujocoArmEnv:
    r"""An $n$-link ball-joint arm in MuJoCo, exposed as $(q,\omega)$.

    Parameters
    ----------
    spec        : :class:`ArmSpec` — geometry, mass, damping, gain, gravity
    dt          : **outer** step, the interval between stored samples. Defaults
                  to 0.05 to match ``windy_arm_nlink_so3.WindyArmNLinkSO3``.
    max_speed   : rejection threshold on $\lVert\omega\rVert_\infty$, mirroring
                  the analytic env so both datasets cover the same regime.

    The MuJoCo integrator runs at ``spec.timestep`` internally; one call to
    :meth:`step` advances by ``dt`` using ``round(dt / timestep)`` MuJoCo steps
    with the torque held constant, which is the same zero-order hold the
    analytic env applies across its Lie–Heun substeps.
    """

    def __init__(self, spec: ArmSpec = ArmSpec(), dt: float = 0.05,
                 max_speed: float = 8.0):
        self.spec = spec
        self.n = spec.n
        self.model, self.data = compile_spec(spec)
        self.dt = float(dt)
        self.max_speed = float(max_speed)

        self.n_substeps = int(round(self.dt / spec.timestep))
        if abs(self.n_substeps * spec.timestep - self.dt) > 1e-12:
            raise ValueError(
                f'dt={self.dt} is not an integer multiple of the MuJoCo '
                f'timestep {spec.timestep}')

        if self.model.nv != 3 * self.n:
            raise AssertionError(
                f'nv={self.model.nv} but 3n={3 * self.n}. The model must use '
                f'ball joints -- a hinge arm cannot represent SO(3)^n.')

        self._bid = [_body_id(self.model, f'link{i + 1}') for i in range(self.n)]
        self._qadr = [self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'j{i + 1}')]
            for i in range(self.n)]
        self._vadr = [self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'j{i + 1}')]
            for i in range(self.n)]

        self.params = params_from_mjmodel(self.model, spec)
        self.t = 0.0

    # ────────────────────────── state I/O ──────────────────────────

    def rotations(self) -> np.ndarray:
        r"""Absolute rotations $R\in\mathbb{R}^{n\times3\times3}$ (world←body)."""
        return np.stack([self.data.xmat[b].reshape(3, 3).copy()
                         for b in self._bid])

    def body_rates(self, check: bool = False, atol: float = 1e-7) -> np.ndarray:
        r"""Absolute body rates $\omega\in\mathbb{R}^{n\times3}$.

        Computed by the forward recursion
        $\omega_i = \Omega_i + R_i^\top R_{i-1}\omega_{i-1}$ from ``qvel``.
        With ``check=True`` this is cross-validated against MuJoCo's own
        ``mj_objectVelocity``; see the module docstring for why.

        The two routes are algebraically identical but numerically distinct —
        ours recurses over $R_i^\top R_{i-1}$, MuJoCo accumulates its com-based
        ``cvel`` down the tree — so they agree to roundoff (~$10^{-9}$ in
        float64), not exactly. ``atol`` is set well above that but far below any
        real convention error, which would show up as an $O(1)$ discrepancy.
        """
        R = self.rotations()
        omega = np.zeros((self.n, 3))
        for i in range(self.n):
            Om = self.data.qvel[self._vadr[i]:self._vadr[i] + 3]
            if i == 0:
                omega[i] = Om
            else:
                omega[i] = Om + R[i].T @ R[i - 1] @ omega[i - 1]

        if check:
            ref = self.body_rates_mujoco()
            err = np.abs(omega - ref).max()
            if err > atol:
                raise AssertionError(
                    f'body-rate cross-check failed: |recursion - '
                    f'mj_objectVelocity|_inf = {err:.3e} > {atol:g}. Either '
                    f'ball-joint qvel is no longer the relative rate in the '
                    f'child frame, or the inertial-frame rotation body_iquat '
                    f'is no longer being undone. Arbitrate with finite '
                    f'differences of xmat: omega = vee(R^T dR/dt).')
        return omega

    def body_rates_mujoco(self) -> np.ndarray:
        r"""Same $\omega$, but straight from ``mj_objectVelocity``.

        ``mj_objectVelocity`` returns a spatial velocity ordered
        ``[angular(3); linear(3)]``, and with ``flg_local=1`` it is expressed in
        the body's **inertial** (principal-axis) frame — *not* the body frame
        that ``xmat`` describes. The two differ by the constant rotation

        .. math::
            Q_i = \mathrm{mat}(\texttt{body\_iquat}_i)
                = R_{\text{body}\leftarrow\text{inertial}}
                = \texttt{xmat}_i^\top\,\texttt{ximat}_i ,

        so the body-frame rate our convention wants is $\omega_i = Q_i\,\omega_i^{\mathcal I}$.

        .. note::
           For these capsules MuJoCo picks ``body_iquat = (0,1,0,0)``, a π
           rotation about $x$, i.e. $Q=\mathrm{diag}(1,-1,-1)$. Skipping the
           correction therefore flips the sign of $\omega_y$ and $\omega_z$ —
           a corruption that leaves $\lVert\omega\rVert$, the kinetic energy and
           every rotationally symmetric diagnostic completely unchanged, so it
           would not show up until the learned model quietly failed. The
           degeneracy $\mathbb{I}=\mathrm{diag}(a,a,b)$ also hides it from
           :func:`params_from_mjmodel`, since $Q\,\mathrm{diag}(a,a,b)\,Q^\top
           = \mathrm{diag}(a,a,b)$.
        """
        out = np.zeros((self.n, 3))
        res = np.zeros(6)
        for i, b in enumerate(self._bid):
            mujoco.mj_objectVelocity(
                self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, b, res, 1)
            Q = np.zeros(9)
            mujoco.mju_quat2Mat(Q, self.model.body_iquat[b])
            out[i] = Q.reshape(3, 3) @ res[:3]
        return out

    def get_state(self, check: bool = False) -> np.ndarray:
        r"""The flat $(12n,)$ state $\big[\mathrm{vec}(R)\,\big|\,\omega\big]$."""
        return np.concatenate([self.rotations().reshape(-1),
                               self.body_rates(check=check).reshape(-1)])

    def set_state(self, R: np.ndarray, omega: np.ndarray) -> None:
        r"""Write an absolute $(R,\omega)$ into MuJoCo's relative coordinates.

        .. math::
            \texttt{qpos}_i = \mathrm{quat}\big(R_{i-1}^\top R_i\big),
            \qquad \texttt{qvel} = \Omega = T(q)\,\omega

        with $R_0 = I$. The rotation is orthonormalised first: MuJoCo stores a
        unit quaternion, so any drift in $R$ would be silently projected away
        and the read-back would not match what was written.
        """
        R = np.asarray(R, dtype=np.float64).reshape(self.n, 3, 3)
        omega = np.asarray(omega, dtype=np.float64).reshape(self.n, 3)
        R = np.stack([_orthonormalise(Ri) for Ri in R])

        for i in range(self.n):
            R_rel = R[i] if i == 0 else R[i - 1].T @ R[i]
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, R_rel.reshape(-1))
            self.data.qpos[self._qadr[i]:self._qadr[i] + 4] = quat

            Om = omega[i] if i == 0 else omega[i] - R[i].T @ R[i - 1] @ omega[i - 1]
            self.data.qvel[self._vadr[i]:self._vadr[i] + 3] = Om

        mujoco.mj_forward(self.model, self.data)

    # ────────────────────────── dynamics ──────────────────────────

    def reset(self, R: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """Place the arm at $(R,\\omega)$ and return the flat state."""
        mujoco.mj_resetData(self.model, self.data)
        self.set_state(R, omega)
        self.t = 0.0
        return self.get_state()

    def step(self, u: np.ndarray) -> np.ndarray:
        r"""Advance one outer ``dt`` with torque $u\in\mathbb{R}^{3n}$ held fixed.

        The delivered joint torque is $\Gamma u$ with
        $\Gamma=\mathrm{blkdiag}(\mathrm{diag}(\gamma_i))$, because each ball
        joint carries three ``<motor>`` actuators geared along $e_x,e_y,e_z$.
        MuJoCo applies the reaction to the parent link automatically, which is
        the $-R_i^\top R_{i+1}u_{i+1}$ term of $g(q)=T(q)^\top\Gamma$.
        """
        self.data.ctrl[:] = np.asarray(u, dtype=np.float64).reshape(-1)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self.t += self.dt
        return self.get_state()

    def close(self) -> None:      # API parity with the analytic env
        pass

    # ────────────────────────── diagnostics ──────────────────────────

    def mass_matrix_relative(self) -> np.ndarray:
        r"""MuJoCo's $M_{\text{rel}}$ (``qM``) densified, $(3n,3n)$.

        .. warning::
           This is in **relative** (joint) coordinates. Ours is in **absolute**
           coordinates. They are congruent, not equal:

           .. math::
               M_{\text{rel}} = T(q)^{-\top} M_{\text{abs}}(q)\, T(q)^{-1}

           Comparing ``qM`` to $M_\theta$ directly is the single easiest way to
           convince yourself the model is wrong when it is fine.
        """
        M = np.zeros((self.model.nv, self.model.nv))
        try:
            # MuJoCo >= 3.11: the packed inertia moved from `data.qM` to
            # `data.M`, and the signature became (model, data, dst) -- note the
            # destination moved from second to third.
            mujoco.mj_fullM(self.model, self.data, M)
        except TypeError:
            mujoco.mj_fullM(self.model, M, self.data.qM)   # <= 3.10
        return M


def _orthonormalise(R: np.ndarray) -> np.ndarray:
    """Nearest rotation to `R` in Frobenius norm, via SVD (det = +1 enforced)."""
    U, _, Vt = np.linalg.svd(R)
    S = np.eye(3)
    S[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ S @ Vt


def random_state(rng: np.random.Generator, n: int, *,
                 omega_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    r"""Sample $(R,\omega)$ from **exactly** the analytic env's reset distribution.

    $R_i$ uniform on $SO(3)$ (Shoemake), $\omega_i\sim U(-1,1)^3$ — see
    ``windy_arm_nlink_so3._random_rotations`` and ``.reset``. Matching this
    matters: the MuJoCo and analytic datasets must cover the same region of
    state space, or a performance difference between them would just be a
    difference in what was sampled.

    Absolute rotations are drawn independently per link, not composed along the
    chain, so the arm starts genuinely folded rather than near-straight.
    """
    R = _random_rotations(rng, n)
    omega = rng.uniform(-omega_scale, omega_scale, size=(n, 3))
    return R, omega


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


if __name__ == '__main__':
    env = MujocoArmEnv()
    rng = np.random.default_rng(0)
    R0, w0 = random_state(rng, env.n)
    env.reset(R0, w0)
    print(f'nq={env.model.nq} nv={env.model.nv} nu={env.model.nu} '
          f'substeps/dt = {env.n_substeps}')
    print(f'omega recursion  = {env.body_rates().reshape(-1)}')
    print(f'omega mj_objectV = {env.body_rates_mujoco().reshape(-1)}')
    print(f'round-trip |dR|  = '
          f'{np.abs(env.rotations() - R0).max():.3e}')
    print(f'round-trip |dw|  = '
          f'{np.abs(env.body_rates() - w0).max():.3e}')
    for _ in range(5):
        env.step(np.zeros(3 * env.n))
    print(f'after 5 steps, omega = {env.body_rates(check=True).reshape(-1)}')
