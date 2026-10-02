"""Dataset generator using gym-pybullet-drones instead of our simulator.

Same 22-dim layout as windy_quadrotor_datagen.py so both trainers load it
unchanged:  [ x_w(3) | vec(R)(9) | v_b(3) | omega_b(3) | u(4) ]

Source: CtrlAviary, Physics.DYN (their explicit rigid-body Euler), real
Crazyflie CF2X from cf2x.urdf, excited by their own DSLPIDControl flying to
random waypoints.

TWO conversions are needed to match our convention:

  * their `vel` is WORLD frame -> we store v_b = R^T v_world
    (their rpy_rates is already body frame, so omega needs no change);

  * u is NORMALISED by the hover input, u = rpm^2 / rpm_hover^2, so u = 1 at
    hover instead of ~2.1e8.  With the raw CF2X numbers (kf = 3.16e-10,
    rpm_hover = 14468) the control column would span 8 orders of magnitude and
    the true mixer entries would be ~1e-10 -- unlearnable in fp32.  The
    normalisation is a pure change of units: the ground-truth mixer simply
    picks up a factor rpm_hover^2, and G_true[2,:] becomes m g / 4.

NOTE their Physics.DYN has NO damping and no drag, so the true D is exactly
ZERO here.  D can therefore only be shown to be near zero, not "recovered".
"""
from __future__ import annotations
import os, sys, pickle
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", "..", ".."))
GPD = os.path.join(ROOT, "third_party", "gym-pybullet-drones")
for p in (ROOT, GPD):
    if p not in sys.path:
        sys.path.insert(0, p)

import pybullet as pb                                                 # noqa: E402
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary            # noqa: E402
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl   # noqa: E402
from gym_pybullet_drones.utils.enums import DroneModel, Physics       # noqa: E402

M, ARM, KF, KM = 0.027, 0.0397, 3.16e-10, 7.94e-12
JD = (1.4e-5, 1.4e-5, 2.17e-5)
G_ACC = 9.8
RPM_HOVER = np.sqrt(M * G_ACC / (4 * KF))
STATE_DIM = 22


def true_G_normalised():
    """Mixer for the NORMALISED input u = rpm^2 / rpm_hover^2."""
    s = RPM_HOVER ** 2
    a = ARM / np.sqrt(2.0)
    G = np.zeros((6, 4))
    G[2, :] = KF * s
    G[3, :] = np.array([-1, -1, 1, 1]) * a * KF * s
    G[4, :] = np.array([-1, 1, 1, -1]) * a * KF * s
    G[5, :] = np.array([-1, 1, -1, 1]) * KM * s
    return G


def sample(seed=0, timesteps=40, trials=96, freq=240, box=0.5):
    rng = np.random.default_rng(seed)
    trajs = []
    for i in range(trials):
        x0 = rng.uniform(-box, box, 3) + np.array([0, 0, 1.0])
        rpy0 = rng.uniform(-0.3, 0.3, 3)
        tgt = x0 + rng.uniform(-box, box, 3)
        tyaw = float(rng.uniform(-np.pi, np.pi))
        env = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=1,
                         initial_xyzs=x0.reshape(1, 3), initial_rpys=rpy0.reshape(1, 3),
                         physics=Physics.DYN, pyb_freq=freq, ctrl_freq=freq,
                         gui=False, record=False, obstacles=False, user_debug_gui=False)
        ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
        obs, _ = env.reset()
        tr = []
        for k in range(timesteps):
            s = obs[0]
            rpm, _, _ = ctrl.computeControlFromState(
                control_timestep=1.0 / freq, state=s,
                target_pos=tgt, target_rpy=np.array([0., 0., tyaw]))
            R = np.array(pb.getMatrixFromQuaternion(s[3:7])).reshape(3, 3)
            v_b = R.T @ s[10:13]                       # world -> body
            u = (np.asarray(rpm, dtype=np.float64) ** 2) / RPM_HOVER ** 2
            tr.append(np.concatenate([s[0:3], R.reshape(9), v_b, s[13:16], u]))
            obs, *_ = env.step(np.asarray(rpm).reshape(1, 4))
        env.close()
        trajs.append(np.stack(tr))
    return np.transpose(np.stack(trajs), (1, 0, 2)), np.arange(timesteps) / freq


def get_dataset(seed=0, samples=96, timesteps=40, save_dir=None, freq=240,
                test_split=0.5, regenerate=False, **kw):
    os.makedirs(save_dir, exist_ok=True)
    tag = f"pybullet_cf2x_s{seed}_n{samples}_T{timesteps}_f{freq}_pid.pkl"
    out = os.path.join(save_dir, tag)
    if os.path.isfile(out) and not regenerate:
        print(f"Loading cached dataset: {out}")
        return pickle.load(open(out, "rb")), out
    print(f"Generating PyBullet CF2X dataset -> {out}")
    trajs, tspan = sample(seed, timesteps, samples, freq)
    allx = trajs[None]                                   # (1, T, N, 22)
    sp = int(samples * (1 - test_split))
    data = {"t": tspan, "x": allx[:, :, :sp, :], "test_x": allx[:, :, sp:, :],
            "test_x_noisy": allx[:, :, sp:, :],
            "settings": dict(source="gym-pybullet-drones CtrlAviary Physics.DYN",
                             drone="CF2X", m=M, J_diag=JD, arm=ARM, kf=KF, km=KM,
                             g=G_ACC, rpm_hover=RPM_HOVER, freq=freq,
                             u_normalised_by="rpm_hover^2", damping=0.0,
                             dt=1.0 / freq, u_mode="dslpid", seed=seed,
                             samples=samples, timesteps=timesteps)}
    pickle.dump(data, open(out, "wb"))
    return data, out


def arrange_data(x, t, num_points=2):
    assert 2 <= num_points <= len(t)
    st = [x[:, i:(-num_points + i + 1) or None, :, :] for i in range(num_points)]
    st = np.stack(st, axis=1)
    return np.reshape(st, (x.shape[0], num_points, -1, x.shape[3])), t[:num_points]


if __name__ == "__main__":
    d, p = get_dataset(save_dir=os.path.join(ROOT, "datasets", "pybullet_quadrotor"),
                       regenerate=True)
    f = d["x"].reshape(-1, STATE_DIM)
    print(f"\ntrain {d['x'].shape}   test {d['test_x'].shape}")
    for nm, sl in (("x_w", slice(0, 3)), ("v_b", slice(12, 15)),
                   ("omega", slice(15, 18)), ("u", slice(18, 22))):
        a = f[:, sl]
        print(f"  {nm:6} range [{a.min():+9.4f}, {a.max():+9.4f}]  std {a.std():8.4f}")
    R = f[:, 3:12].reshape(-1, 3, 3)
    print(f"  max |R R^T - I| = {np.abs(np.einsum('nij,nkj->nik',R,R)-np.eye(3)).max():.2e}")
    print(f"\n  rpm_hover = {RPM_HOVER:.1f}   true G (normalised u):")
    Gt = true_G_normalised()
    for r, nm in enumerate(("F_x","F_y","F_z","tau_x","tau_y","tau_z")):
        print(f"    {nm:6} [{' '.join(f'{v:+.5f}' for v in Gt[r])}]")
    print(f"  true M^-1 diag: [{', '.join(f'{v:.1f}' for v in [1/M]*3 + [1/j for j in JD])}]")
