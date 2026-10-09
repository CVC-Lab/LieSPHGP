"""BlueROV2 Heavy on SE(3) in port-Hamiltonian form: vehicle constants, thruster model, dynamics, Lie-group integrator.

World frame NED-like (z DOWN, = depth), body frame forward-right-down, as in the Marinarium recordings (checked in
envs/rov_se3_marinarium/README.md) and in Fossen's convention. State: position p (world), R (body -> world),
body twist nu = (v, w). With momentum P = M nu and M = M_RB + M_A (diagonal: CG at the body origin),

    H(p, R, P) = 1/2 P^T M^-1 P + V(p, R),     V = (B - W) z + B e3^T R r_b

    p_dot = R v,   R_dot = R [w]x
    P_dot = J(P) nu  -  D(nu) nu  -  grad V  +  G u  +  noise

where J(P) nu = [P_v x w ; P_v x v + P_w x w] is the rigid-body (Kirchhoff) term that conserves energy, D(nu) is the
damping and G the input map: G = E (6 x 8 thruster allocation) when u are thruster forces, G = I when u is the wrench.

Published parameters: von Benzon et al., "An open-source benchmark simulator: control of a BlueROV2 underwater
robot", J. Mar. Sci. Eng. 10(12), 2022, Table A1 (heavy configuration), as used by the Fossen baseline of
github.com/ViktorNfa/bluerov2_dynamics (fossen/BlueROV2.py). Thruster geometry and motor order: the PX4 rotor list of
that repo (rosbags/create_thrust_torque_csv.py), which matches the motor channels of the recordings.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
T200_TABLE = THIS_DIR / "data" / "t200_thrust_table.csv"
E3 = np.array([0.0, 0.0, 1.0])

# --------------------------------------------------------------------------- published constants (von Benzon 2022, Table A1, heavy)
PUBLISHED = {
    "mass": 13.5,                                       # kg
    "volume": 0.0134,                                   # m^3 displaced
    "rho": 1000.0,                                      # kg/m^3 (fresh-water tank)
    "gravity": 9.82,                                    # m/s^2
    "inertia": [0.26, 0.23, 0.37],                      # kg m^2 about the CG
    "center_of_buoyancy": [0.0, 0.0, -0.01],            # m, CB relative to CG (body frame; -z = above)
    "added_mass": [6.36, 7.12, 18.68, 0.189, 0.135, 0.222],          # -[X_ud, Y_vd, Z_wd, K_pd, M_qd, N_rd]
    "linear_damping": [13.7, 0.0, 33.0, 0.0, 0.8, 0.0],              # -[X_u, Y_v, Z_w, K_p, M_q, N_r]
    "quadratic_damping": [141.0, 217.0, 190.0, 1.19, 0.47, 1.5],     # -[X_u|u|, Y_v|v|, Z_w|w|, K_p|p|, M_q|q|, N_r|r|]
}

# Thrusters in PX4 motor order (u1..u8 of the recordings): unit axis (thrust direction for a positive command) and
# position relative to the body origin (m). Horizontal T1-T4 at 45 deg, vertical T5-T8.
THRUSTERS = [
    ((1.0, -1.0, 0.0), (0.14, 0.10, 0.06)),
    ((1.0, 1.0, 0.0), (0.14, -0.10, 0.06)),
    ((1.0, 1.0, 0.0), (-0.14, 0.10, 0.06)),
    ((1.0, -1.0, 0.0), (-0.14, -0.10, 0.06)),
    ((0.0, 0.0, -1.0), (0.12, 0.22, 0.00)),
    ((0.0, 0.0, 1.0), (0.12, -0.22, 0.00)),
    ((0.0, 0.0, 1.0), (-0.12, 0.22, 0.00)),
    ((0.0, 0.0, -1.0), (-0.12, -0.22, 0.00)),
]


def allocation_matrix() -> np.ndarray:
    """E (6 x 8): wrench [F; tau] = E @ thrust, thrust_i in N along the unit axis of thruster i."""
    E = np.zeros((6, len(THRUSTERS)))
    for i, (axis, pos) in enumerate(THRUSTERS):
        a = np.asarray(axis, float); a /= np.linalg.norm(a)
        E[:3, i], E[3:, i] = a, np.cross(np.asarray(pos, float), a)
    return E


# --------------------------------------------------------------------------- T200 thruster: command <-> thrust
class T200:
    """Blue Robotics T200 static thrust (N) as a function of PWM (us) and battery voltage (V), from the public table.

    PX4 reversible motor command u in [-1, 1] -> PWM = 1500 + 400 u. Bilinear interpolation in (PWM, voltage); the
    voltage is clipped to the table range 10-20 V. Dead band |u| < 0.07; forward thrust is larger than reverse.
    """

    def __init__(self, path: Path = T200_TABLE):
        rows = [r for r in csv.reader(line for line in path.open() if not line.startswith("#"))][1:]
        values = np.asarray(rows, float)
        self.voltages = np.unique(values[:, 0]); self.pwm = np.unique(values[:, 1])
        self.thrust = np.zeros((self.pwm.size, self.voltages.size))
        for v, p, f in values:
            self.thrust[np.searchsorted(self.pwm, p), np.searchsorted(self.voltages, v)] = f

    def _column(self, voltage: float) -> np.ndarray:
        v = float(np.clip(voltage, self.voltages[0], self.voltages[-1]))
        j = int(np.clip(np.searchsorted(self.voltages, v) - 1, 0, self.voltages.size - 2))
        w = (v - self.voltages[j]) / (self.voltages[j + 1] - self.voltages[j])
        return (1 - w) * self.thrust[:, j] + w * self.thrust[:, j + 1]

    def thrust_from_command(self, command: np.ndarray, voltage) -> np.ndarray:
        """Thrust (N) for commands in [-1, 1]; NaN (motor not commanded) -> 0. voltage: scalar or one per row."""
        command = np.nan_to_num(np.asarray(command, float), nan=0.0)
        pwm = 1500.0 + 400.0 * np.clip(command, -1.0, 1.0)
        volts = np.broadcast_to(np.asarray(voltage, float), command.shape[:1] if command.ndim > 1 else ())
        if command.ndim == 1:
            return np.interp(pwm, self.pwm, self._column(float(volts)))
        out = np.empty_like(pwm)
        for k in range(command.shape[0]):
            out[k] = np.interp(pwm[k], self.pwm, self._column(float(volts[k])))
        return out

    def command_from_thrust(self, thrust: np.ndarray, voltage: float) -> np.ndarray:
        """Inverse map, one branch per sign: thrust > 0 -> PWM above 1500, thrust < 0 -> below, thrust 0 -> command 0
        exactly (the dead band is not inverted to its edge). The table is not strictly monotone at some voltages (flat
        dead band; a few entries near saturation dip, e.g. 12 V at 1892-1896 us), so each branch is inverted on its
        running-max envelope of |thrust|; thrust beyond the table range clips to a command of +-1."""
        column = self._column(voltage)
        thrust = np.asarray(thrust, float)
        pos, neg = self.pwm >= 1500.0, self.pwm <= 1500.0
        up_f, up_pwm = np.maximum.accumulate(np.maximum(column[pos], 0.0)), self.pwm[pos]                 # PWM rising from 1500
        dn_f, dn_pwm = np.maximum.accumulate(np.maximum(-column[neg][::-1], 0.0)), self.pwm[neg][::-1]   # PWM falling from 1500
        pwm = np.where(thrust > 0, np.interp(thrust, up_f, up_pwm),
                       np.where(thrust < 0, np.interp(-thrust, dn_f, dn_pwm), 1500.0))
        return (pwm - 1500.0) / 400.0

    def limits(self, voltage: float) -> tuple[float, float]:
        column = self._column(voltage)
        return float(column.min()), float(column.max())


# --------------------------------------------------------------------------- SO(3) helpers
def hat(w: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def exp_so3(phi: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(phi))
    K = hat(phi)
    if theta < 1e-8:
        return np.eye(3) + K + 0.5 * K @ K
    return np.eye(3) + np.sin(theta) / theta * K + (1 - np.cos(theta)) / theta**2 * K @ K


# --------------------------------------------------------------------------- vehicle
@dataclass
class BlueROV2Params:
    mass: float = PUBLISHED["mass"]
    volume: float = PUBLISHED["volume"]
    rho: float = PUBLISHED["rho"]
    gravity: float = PUBLISHED["gravity"]
    inertia: list = field(default_factory=lambda: list(PUBLISHED["inertia"]))
    center_of_buoyancy: list = field(default_factory=lambda: list(PUBLISHED["center_of_buoyancy"]))
    added_mass: list = field(default_factory=lambda: list(PUBLISHED["added_mass"]))
    linear_damping: list = field(default_factory=lambda: list(PUBLISHED["linear_damping"]))
    quadratic_damping: list = field(default_factory=lambda: list(PUBLISHED["quadratic_damping"]))

    @property
    def weight(self) -> float:
        return self.mass * self.gravity

    @property
    def buoyancy(self) -> float:
        return self.rho * self.gravity * self.volume

    @property
    def M_diag(self) -> np.ndarray:
        return np.r_[np.full(3, self.mass), self.inertia] + np.asarray(self.added_mass, float)


class BlueROV2:
    """Deterministic + stochastic BlueROV2 Heavy dynamics on SE(3) (body twist form) with a Lie-group Heun step.

    dissipation[ch] in {none, linear, quadratic, linear_quadratic, constant} per channel (linear: v, angular: w):
        D(nu) nu = (D_L + D_Q |nu|) nu with the parts switched off as requested;
        constant (as the quadrotor generator): D_L = c m (linear), c J (angular, rigid-body inertia), D_Q = 0, with
        c = constant_c[ch] (1/s).
    The generalized force `tau` passed to `nu_dot` / `heun_step` already includes the actuators and any disturbance.
    """

    def __init__(self, params: BlueROV2Params | None = None, dissipation: dict | None = None,
                 constant_c: dict | None = None):
        self.p = params or BlueROV2Params()
        self.M = self.p.M_diag
        self.M_inv = 1.0 / self.M
        self.r_b = np.asarray(self.p.center_of_buoyancy, float)
        dissipation = dissipation or {"linear": "linear_quadratic", "angular": "linear_quadratic"}
        D_L, D_Q = np.asarray(self.p.linear_damping, float), np.asarray(self.p.quadratic_damping, float)
        self.D_L, self.D_Q = np.zeros(6), np.zeros(6)
        for ch, sl in (("linear", slice(0, 3)), ("angular", slice(3, 6))):
            kind = dissipation[ch]
            if kind not in ("none", "linear", "quadratic", "linear_quadratic", "constant"):
                raise ValueError(f"dissipation.{ch}: unknown type {kind!r}")
            if kind == "constant":
                if constant_c is None or constant_c.get(ch) is None:
                    raise ValueError(f"dissipation.{ch}: constant needs a coefficient c")
                rigid = np.full(3, self.p.mass) if ch == "linear" else np.asarray(self.p.inertia, float)
                self.D_L[sl] = float(constant_c[ch]) * rigid
            if kind in ("linear", "linear_quadratic"):
                self.D_L[sl] = D_L[sl]
            if kind in ("quadratic", "linear_quadratic"):
                self.D_Q[sl] = D_Q[sl]
        self.E = allocation_matrix()

    # -- forces ---------------------------------------------------------------
    def restoring(self, R: np.ndarray) -> np.ndarray:
        """Generalized hydrostatic force on the body = -grad V, V = (B - W) z + B e3^T R r_b (z down; weight at the CG =
        body origin, buoyancy at r_b): force from (W-B) along world down; torque r_b x buoyancy."""
        down_body = R.T @ E3
        force = (self.p.weight - self.p.buoyancy) * down_body
        torque = np.cross(self.r_b, -self.p.buoyancy * down_body)
        return np.r_[force, torque]

    def damping_force(self, nu: np.ndarray) -> np.ndarray:
        return -(self.D_L + self.D_Q * np.abs(nu)) * nu

    def coriolis_force(self, nu: np.ndarray) -> np.ndarray:
        """J(P) nu: rigid-body + added-mass Coriolis/centripetal force, energy conserving (nu . result = 0)."""
        v, w = nu[:3], nu[3:]
        P_v, P_w = self.M[:3] * v, self.M[3:] * w
        return np.r_[np.cross(P_v, w), np.cross(P_v, v) + np.cross(P_w, w)]

    def nu_dot(self, R: np.ndarray, nu: np.ndarray, tau: np.ndarray) -> np.ndarray:
        return self.M_inv * (self.coriolis_force(nu) + self.damping_force(nu) + self.restoring(R) + tau)

    # -- integrator -----------------------------------------------------------
    def heun_step(self, position, R, nu, tau, h, noise_impulse=None):
        """One Lie-group Heun step of length h (second order in the drift). tau is held over the step;
        noise_impulse (6,) is a generalized impulse added to M nu (already integrated over h)."""
        dn = np.zeros(6) if noise_impulse is None else self.M_inv * noise_impulse
        a1 = self.nu_dot(R, nu, tau)
        R1 = R @ exp_so3(nu[3:] * h)
        nu1 = nu + a1 * h + dn
        p_dot1 = R @ nu[:3]
        a2 = self.nu_dot(R1, nu1, tau)
        p_dot2 = R1 @ nu1[:3]
        R_new = R @ exp_so3(0.5 * (nu[3:] + nu1[3:]) * h)
        nu_new = nu + 0.5 * (a1 + a2) * h + dn
        return position + 0.5 * (p_dot1 + p_dot2) * h, R_new, nu_new
