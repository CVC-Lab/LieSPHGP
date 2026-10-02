r"""MJCF generation for an $n$-link **ball-joint** arm, plus exact extraction of
our :class:`ArmParams` from the compiled MuJoCo model.

Why ball joints
---------------
The port-Hamiltonian model in ``utils/ph_network_nlink.py`` carries a *full*
rotation per link, $q=(R_1,\dots,R_n)\in SO(3)^n$, and a $3n$-dimensional body
rate $\omega$. That is $3n$ degrees of freedom, i.e. one **spherical** joint per
link. A hinge (revolute) arm has only $n$ DOF; handed to this model it would

* leave $3n-n$ directions of $M(q)$ completely unexcited (unidentifiable), and
* immediately leave the constraint manifold, because the Lie–Heun integrator in
  ``utils/lie_integrator_nlink.py`` carries no constraint forces.

So the MuJoCo model must use ``<joint type="ball"/>``. Then
$n_v = 3n$ exactly matches $\omega$.

Why capsules and not the analytic default
-----------------------------------------
:func:`arm_nlink_physics.uniform_chain_params` uses
$\mathbb{I}_i=\mathrm{diag}(0,0,\text{scale}\cdot mL^2)$ — **two zero principal
moments**. MuJoCo rejects that (it requires a positive-definite inertia), so the
links here are solid capsules of radius $r$ about the body $z$-axis:

.. math::
    \mathbb{I}_{zz} = \tfrac12 m r^2, \qquad
    \mathbb{I}_{xx} = \mathbb{I}_{yy} = \tfrac{1}{12}m\,(3r^2 + L^2)

Both are positive for $r>0$. The radius is a genuine **conditioning knob**: the
axial direction of $M$ is the light one, and

.. math::
    \mathrm{cond}(M) \;\approx\;
        \frac{\mathbb{I}_{xx} + m\lVert c\rVert^2}{\mathbb{I}_{zz}}
      = \frac{\tfrac{1}{12}(3r^2+L^2) + \tfrac14 L^2}{\tfrac12 r^2}

which blows up as $r\to0$. At $L=1$: $r=0.05$ gives $\mathrm{cond}\approx270$,
$r=0.3$ gives $\approx 7.9$. The default $r=0.3$ is a stubby but perfectly
physical link that keeps $M^{-1}$ well behaved. This is the same failure mode
the analytic env warns about in
:func:`arm_nlink_physics.uniform_chain_params` — a point mass on its own axis
has no inertia about that axis — only here it is cured with real geometry
instead of a hand-set ``inertia_scale``.

Ground truth flows *out* of MuJoCo, not into it
-----------------------------------------------
We never hand-write $\mathbb{I}$. The XML declares geometry and mass; MuJoCo's
compiler computes the inertial properties; :func:`params_from_mjmodel` then
reads $(m,\ell,c,\mathbb{I},d,\Gamma,g)$ back out of the compiled ``mjModel``.
That makes the analytic :class:`ArmParams` a *consequence* of the MuJoCo model
rather than a parallel hand-maintained copy that can silently drift.
"""
from __future__ import annotations

import os
import sys
from typing import NamedTuple, Optional, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mujoco                                                   # noqa: E402

from arm_nlink_physics import ArmParams                          # noqa: E402


# ══════════════════════════════════════════════════════════════════════
# Specification
# ══════════════════════════════════════════════════════════════════════

class ArmSpec(NamedTuple):
    r"""Everything needed to emit an MJCF and reproduce it exactly.

    Fields
    ------
    n            : number of links
    link_length  : $L$, joint$\to$joint offset along body $+e_z$
    link_radius  : $r$, capsule radius. See the module docstring — this is the
                   conditioning knob for $M(q)$.
    mass         : $m_i$, per link (MuJoCo rescales the geom inertia to hit it)
    damping      : $d_i$, joint viscous friction on the **relative** rate
    gain         : $\gamma_i=(\gamma_x,\gamma_y,\gamma_z)$, per-axis actuator
                   gain, so $g(q)=T(q)^\top\Gamma$ with
                   $\Gamma=\mathrm{blkdiag}(\mathrm{diag}(\gamma_i))$
    gravity      : $g$ (positive; MuJoCo gets $-g\,e_z$)
    timestep     : MuJoCo internal integrator step
    armature     : rotor inertia added to the **joint-space** diagonal.
                   *Outside our model class* — see :mod:`README`. 0 = matched.
    frictionloss : Coulomb joint friction. Non-smooth, also outside the model
                   class ($D$ is linear viscous). 0 = matched.
    """
    n:            int = 2
    link_length:  float = 1.0
    link_radius:  float = 0.3
    mass:         float = 1.0
    damping:      float = 0.5
    gain:         Sequence[float] = (0.5, 0.7, 0.5)
    gravity:      float = 9.81
    timestep:     float = 0.001
    armature:     float = 0.0
    frictionloss: float = 0.0

    @property
    def matched(self) -> bool:
        """True when every effect outside our model class is switched off."""
        return self.armature == 0.0 and self.frictionloss == 0.0

    def tag(self) -> str:
        """Short filename-safe identity string."""
        gain = '-'.join(f'{v:g}'.replace('.', 'p') for v in self.gain)
        base = (f'arm{self.n}ball_L{self.link_length:g}_r{self.link_radius:g}'
                f'_m{self.mass:g}_d{self.damping:g}_G{gain}')
        if not self.matched:
            base += (f'_arm{self.armature:g}'.replace('.', 'p')
                     + f'_fl{self.frictionloss:g}'.replace('.', 'p'))
        return base.replace('.', 'p').replace('-0p', '-0.')  # keep gain dashes


# ══════════════════════════════════════════════════════════════════════
# MJCF emission
# ══════════════════════════════════════════════════════════════════════

def build_mjcf(spec: ArmSpec) -> str:
    r"""Return the MJCF XML string for `spec`.

    Structure — link $i$ is a child of link $i-1$, offset by $\ell_{i-1}=L e_z$:

    .. code::

        world
         └ link1  (pos 0 0 0)      ball joint j1 at origin
            └ link2  (pos 0 0 L)   ball joint j2 at link1's tip
               └ ...

    At $R_i=I$ every link points along $+e_z$, i.e. straight **up**. That is the
    inverted (unstable) equilibrium, matching the analytic env's convention
    $V(q)=g\sum_j\big[R_j(\mu_j^{>}\ell_j+m_jc_j)\big]_z$, which is maximal
    there.

    Actuation: three ``<motor>`` per ball joint, with ``gear`` along $e_x,e_y,e_z$
    scaled by $\gamma$. Applying control $u_i\in\mathbb{R}^3$ therefore delivers
    joint torque $\mathrm{diag}(\gamma_i)u_i$, and MuJoCo automatically applies
    the equal-and-opposite reaction to the parent link — which is exactly the
    $-R_i^\top R_{i+1}u_{i+1}$ term in $g(q)=T(q)^\top\Gamma$.

    Contacts are disabled globally: our model has no contact term, and a
    self-colliding arm would be unlearnable by construction.
    """
    L, r, m = spec.link_length, spec.link_radius, spec.mass
    jopts = (f'damping="{spec.damping}" armature="{spec.armature}" '
             f'frictionloss="{spec.frictionloss}"')

    # Build the nested <body> chain from the inside out.
    body = ''
    for i in range(spec.n - 1, -1, -1):
        pos = '0 0 0' if i == 0 else f'0 0 {L}'
        inner = _indent(body, 2) if body else ''
        body = (
            f'<body name="link{i + 1}" pos="{pos}">\n'
            f'  <joint name="j{i + 1}" type="ball" pos="0 0 0" {jopts}/>\n'
            f'  <geom name="g{i + 1}" type="capsule" fromto="0 0 0 0 0 {L}"\n'
            f'        size="{r}" mass="{m}" contype="0" conaffinity="0"\n'
            f'        rgba="{0.35 + 0.25 * i:.2f} 0.55 0.85 1"/>\n'
            f'{inner}'
            f'</body>\n'
        )

    motors = ''
    for i in range(spec.n):
        for k, gk in enumerate(spec.gain):
            axis = ['0', '0', '0']
            axis[k] = f'{gk}'
            motors += (f'    <motor name="u{i + 1}{"xyz"[k]}" joint="j{i + 1}" '
                       f'gear="{" ".join(axis)}"/>\n')

    return (
        f'<mujoco model="arm{spec.n}link_ball">\n'
        f'  <compiler angle="radian" inertiafromgeom="true"/>\n'
        f'  <option timestep="{spec.timestep}" integrator="RK4"\n'
        f'          gravity="0 0 {-spec.gravity}">\n'
        f'    <flag contact="disable"/>\n'
        f'  </option>\n'
        f'  <worldbody>\n'
        f'    <light pos="0 0 4" dir="0 0 -1"/>\n'
        f'{_indent(body, 4)}'
        f'  </worldbody>\n'
        f'  <actuator>\n'
        f'{motors}'
        f'  </actuator>\n'
        f'</mujoco>\n'
    )


def _indent(block: str, k: int) -> str:
    pad = ' ' * k
    return ''.join(pad + ln + '\n' for ln in block.rstrip('\n').split('\n'))


def compile_spec(spec: ArmSpec):
    """`ArmSpec` → `(mjModel, mjData)`."""
    model = mujoco.MjModel.from_xml_string(build_mjcf(spec))
    return model, mujoco.MjData(model)


# ══════════════════════════════════════════════════════════════════════
# MuJoCo → ArmParams
# ══════════════════════════════════════════════════════════════════════

def params_from_mjmodel(model, spec: ArmSpec, dtype=np.float64) -> ArmParams:
    r"""Read the analytic :class:`ArmParams` out of a **compiled** ``mjModel``.

    Every field is taken from MuJoCo, never re-derived, so the analytic closed
    forms and the simulator cannot disagree about what the arm *is*:

    ==================  ================================  ====================
    our field           mjModel source                    note
    ==================  ================================  ====================
    ``m[i]``            ``body_mass``                     kg
    ``c[i]``            ``body_ipos``                     COM in body frame
    ``I_body[i]``       ``body_inertia`` + ``body_iquat`` rotated to body frame
    ``ell[i]``          ``body_pos`` of the *child*       0 for the last link
    ``d[i]``            ``dof_damping``                   isotropic per joint
    ``gain[i]``         ``actuator_gear``                 the diagonal of Γ
    ``g``               ``opt.gravity``                   $-g_z$
    ==================  ================================  ====================

    The inertia needs one rotation. MuJoCo stores $\mathbb{I}$ *diagonal* in a
    principal-axis frame and records that frame as the quaternion
    ``body_iquat``; with $Q=\mathrm{mat}(\texttt{body\_iquat})$ mapping inertial
    coordinates into body coordinates,

    .. math::
        \mathbb{I}_{\text{body}} = Q\,\mathrm{diag}(\texttt{body\_inertia})\,Q^\top .

    Our ``StructuredMass`` carries a **full** $3\times3$ inertia per link (the
    ``9*n`` slice of its core vector), so it can represent any such $Q$ — no
    principal-axis alignment is required of the learner.

    .. note::
       ``armature`` is deliberately **not** returned. It adds $\alpha I$ to the
       *joint-space* mass matrix, i.e. $T^\top\alpha T$ in absolute coordinates,
       which is not of the form
       $\mathrm{blkdiag}(\mathbb{I}_i)+\sum_i m_i J_{v,i}^\top J_{v,i}$. There is
       no ``ArmParams`` that represents it — that is precisely what makes the
       misspecified experiment misspecified.
    """
    n = spec.n
    bid = [_body_id(model, f'link{i + 1}') for i in range(n)]

    m = np.array([model.body_mass[b] for b in bid], dtype=dtype)
    c = np.array([model.body_ipos[b] for b in bid], dtype=dtype)

    I_body = np.zeros((n, 3, 3), dtype=dtype)
    for i, b in enumerate(bid):
        Q = np.zeros(9)
        mujoco.mju_quat2Mat(Q, model.body_iquat[b])
        Q = Q.reshape(3, 3)
        I_body[i] = Q @ np.diag(model.body_inertia[b]) @ Q.T

    # ell[i] is the joint i -> joint i+1 offset, which MuJoCo stores as the
    # child body's position in the parent frame. The last link has no child, so
    # its ell never enters the dynamics -- keep it zero, matching
    # arm_nlink_physics.uniform_chain_params.
    ell = np.zeros((n, 3), dtype=dtype)
    for i in range(n - 1):
        ell[i] = model.body_pos[bid[i + 1]]

    # One ball joint per link contributes 3 consecutive DOFs.
    d = np.array([model.dof_damping[3 * i] for i in range(n)], dtype=dtype)
    for i in range(n):
        block = model.dof_damping[3 * i:3 * i + 3]
        if not np.allclose(block, block[0]):
            raise ValueError(
                f'joint {i} has anisotropic damping {block}; the analytic '
                f'D(q) = T^T blkdiag(d_i I_3) T assumes it is isotropic.')

    # actuator_gear is (nu, 6); for a ball joint only the first 3 are used.
    gain = np.zeros((n, 3), dtype=dtype)
    for a in range(model.nu):
        j = model.actuator_trnid[a, 0]
        i = int(model.jnt_bodyid[j]) - bid[0]
        axis = model.actuator_gear[a, :3]
        k = int(np.argmax(np.abs(axis)))
        gain[i, k] = axis[k]

    return ArmParams(
        m=m,
        ell=ell,
        c=c,
        I_body=I_body,
        d=d,
        kappa=np.zeros(n, dtype=dtype),      # no absolute air drag in MuJoCo
        a=np.zeros(n, dtype=dtype),          # no wind field in MuJoCo
        g=np.asarray(-model.opt.gravity[2], dtype=dtype),
        gain=gain,
        varying_friction=np.asarray(0.0, dtype=dtype),
    )


def _body_id(model, name: str) -> int:
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if b < 0:
        raise KeyError(f'no body named {name!r} in the compiled model')
    return b


if __name__ == '__main__':
    spec = ArmSpec()
    print(build_mjcf(spec))
    model, _ = compile_spec(spec)
    p = params_from_mjmodel(model, spec)
    print(f'nq={model.nq}  nv={model.nv}  nu={model.nu}   (want nv = 3n = {3 * spec.n})')
    print(f'm    = {p.m}')
    print(f'ell  =\n{p.ell}')
    print(f'c    =\n{p.c}')
    print(f'I[0] =\n{p.I_body[0]}')
    print(f'd    = {p.d}')
    print(f'gain =\n{p.gain}')
    print(f'g    = {p.g}')
