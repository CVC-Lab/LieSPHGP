"""
Tennis Racket Phase Space Visualization

Generates TWO output figures per dataset:

  Figure 1 — Principal-axis angular velocity phase space
     6 rows × num_configs cols
     Per-axis (Euler angle, body angular velocity) projections.
     Axes correspond to body-frame principal axes e1, e2, e3.

  Figure 2 — SO(3)-aware phase space
     8 rows × num_configs cols
     For each of {Train, Test}:
       - e1 principal axis trajectory on S² (3D, orientation of long head axis)
       - Intermediate-axis alignment α = arccos(|ω̂ · e₂|) vs dα/dt (2D phase portrait)
       - ||ω|| over time
       - Rotational kinetic energy T(t) = ½ ωᵀ I ω (no gravity — free body)
"""

import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)


# ─────────────────── I/O ───────────────────

def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# ─────────────────── Geometry helpers ───────────────────

def rotmat_to_euler(R_flat):
    """Vectorized batch rotmat → Euler ZYX (roll, pitch, yaw).
    R_flat: (..., 9) → (..., 3) in radians.
    NOTE: ZYX has gimbal lock at pitch = ±π/2; visualization artifact only.
    """
    Rs = R_flat.reshape(*R_flat.shape[:-1], 3, 3)
    R00 = Rs[..., 0, 0]
    R10 = Rs[..., 1, 0]
    R20 = Rs[..., 2, 0]
    R21 = Rs[..., 2, 1]
    R22 = Rs[..., 2, 2]
    R12 = Rs[..., 1, 2]
    R11 = Rs[..., 1, 1]

    sy = np.sqrt(R00**2 + R10**2)
    near_singular = sy < 1e-6

    roll = np.where(near_singular,
                    np.arctan2(-R12, R11),
                    np.arctan2(R21, R22))
    pitch = np.arctan2(-R20, sy)
    yaw = np.where(near_singular,
                   np.zeros_like(R00),
                   np.arctan2(R10, R00))
    return np.stack([roll, pitch, yaw], axis=-1)


def _rotmat_principal_axis(R_flat, axis=0):
    """Extract a principal body-frame axis direction R @ e_i from flat rotation matrices.
    R_flat: (..., 9) → (..., 3).
    Column i of R (row-major flat) = R_flat[..., [i*3+0, i*3+1, i*3+2]]
    but since R is stored row-major: column i = R_flat[..., [i, i+3, i+6]].
    """
    return R_flat[..., [axis, axis + 3, axis + 6]]


def _resolve_dataset_path(save_dir, filename):
    """Return (filename, file_path), or (None, None) if not found."""
    if filename is None:
        files = sorted([f for f in os.listdir(save_dir) if f.endswith('.pkl')])
        if not files:
            print(f"No .pkl files in {save_dir}")
            return None, None
        filename = files[0]
        print(f"Auto-selected: {filename}")
    file_path = os.path.join(save_dir, filename)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None, None
    return filename, file_path


# ─────────────────── Figure 1: Principal-axis phase space ───────────────────

def plot_racket_phase_space(save_dir, num_trajs_to_plot=15, filename=None):
    """Per-principal-axis (Euler angle, body ω_i) phase portrait.

    Mirrors the original pendulum plot but:
    - axis labels are renamed to e1 (stable/long), e2 (unstable/intermediate), e3 (stable/handle)
    - colormap is keyed on ||disturbance torque|| instead of ||wind force||
    - column titles show 'Config N' (racket geometry) instead of 'Batch N'
    - figure title is updated for the tennis racket context
    """
    filename, file_path = _resolve_dataset_path(save_dir, filename)
    if file_path is None:
        return

    print(f"Loading: {file_path}")
    data = load_data(file_path)

    train_data = data['x']                  # (num_configs, T, N_train, 15)
    test_data  = data.get('test_x', None)   # (num_configs, T, N_test,  15)

    num_configs = train_data.shape[0]
    # CHANGED: axis labels reflect principal body-frame axes of the racket,
    # not generic X/Y/Z pendulum Euler angles.
    axis_labels = ['e1 (stable, long)', 'e2 (unstable)', 'e3 (stable, handle)']

    # Global disturbance-torque range for shared colormap normalization.
    # CHANGED: the "action" stored in dims 12:15 is the constant disturbance
    # torque (Nm), not a wind force, so label and semantics are updated.
    all_u = [np.linalg.norm(train_data[..., 12:15], axis=-1).flatten()]
    if test_data is not None:
        all_u.append(np.linalg.norm(test_data[..., 12:15], axis=-1).flatten())
    u_max = np.concatenate(all_u).max()
    norm = Normalize(vmin=0, vmax=max(u_max, 0.01))
    cmap = plt.get_cmap('viridis')

    nrows = 6
    ncols = num_configs  # CHANGED: variable renamed from num_us → num_configs
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(4.5 * ncols, 3 * nrows), squeeze=False)

    datasets = [
        ("Train", train_data, 0),
        ("Test",  test_data,  3),
    ]

    for ds_label, ds_data, row_offset in datasets:
        if ds_data is None:
            continue
        N_avail = ds_data.shape[2]
        n_plot = min(N_avail, num_trajs_to_plot)

        for cfg_idx in range(num_configs):  # CHANGED: u_idx → cfg_idx
            batch = ds_data[cfg_idx]              # (T, N, 15)
            R_all = batch[..., :9]                # (T, N, 9)
            eulers_all = rotmat_to_euler(R_all)   # (T, N, 3)

            for axis_idx in range(3):
                ax = axes[row_offset + axis_idx, cfg_idx]

                for trial in range(n_plot):
                    angle = np.unwrap(eulers_all[:, trial, axis_idx])
                    omega = batch[:, trial, 9 + axis_idx]
                    u_norms = np.linalg.norm(batch[:, trial, 12:15], axis=1)

                    points = np.array([angle, omega]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    lc = LineCollection(segments, cmap=cmap, norm=norm,
                                        alpha=0.5, linewidth=1.0)
                    lc.set_array(u_norms[:-1])
                    ax.add_collection(lc)
                    ax.scatter(angle[0], omega[0], color='black', s=8, zorder=3)

                ax.autoscale()
                ax.grid(True, alpha=0.3, linestyle='--')

                if cfg_idx == 0:
                    ax.set_ylabel(f"{ds_label} — {axis_labels[axis_idx]}\n" +
                                  r"$\omega$ (rad/s)", fontsize=9)

                if row_offset + axis_idx == 0:
                    # CHANGED: 'Batch N' → 'Config N' (each column = racket geometry)
                    ax.set_title(f"Config {cfg_idx}", fontsize=11, fontweight='bold')

                if row_offset + axis_idx == nrows - 1:
                    ax.set_xlabel("Angle (rad)", fontsize=9)
                else:
                    ax.set_xticklabels([])

    # Colorbar
    # CHANGED: label updated from wind force to disturbance torque
    cbar_ax = fig.add_axes([0.93, 0.12, 0.015, 0.76])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cb.set_label(r'$\|\tau_d\|$ (Nm)', fontsize=10)

    # CHANGED: title updated for the tennis racket free rigid body
    fig.suptitle("Tennis Racket Phase Space — Rotation about e1, e2, e3 (Euler ZYX)",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.subplots_adjust(right=0.91, hspace=0.25, wspace=0.3)

    ds_name = os.path.splitext(filename)[0]
    # CHANGED: output filename suffix from _phase_space_XYZ → _phase_space_principal
    out_path = os.path.join(save_dir, f'{ds_name}_phase_space_principal.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


# ─────────────────── Figure 2: SO(3)-aware phase space ───────────────────

def plot_so3_phase_space(save_dir, num_trajs_to_plot=15, filename=None):
    """Geometrically-natural visualizations for SO(3) tennis-racket data.

    Per dataset (train, test) and per geometry config:
      Row 0/4: e1 principal axis trajectory on S²  (orientation of long head axis in world frame)
      Row 1/5: Intermediate-axis alignment α = arccos(|ω̂ · e₂|) vs dα/dt  — Dzhanibekov portrait
      Row 2/6: ||ω|| (rad/s) over time
      Row 3/7: Rotational kinetic energy T(t) = ½ ωᵀ I ω  (no gravity — free rigid body)

    Key differences from the pendulum SO(3) plot:
    - No gravity → no potential energy term; Row 3/7 shows kinetic energy only.
    - Bob-on-S² replaced by e1-axis-on-S²: tracks where the long head axis points.
    - Tilt angle replaced by intermediate-axis alignment (Dzhanibekov instability).
    - Colormap keyed on ||disturbance torque|| not ||wind force||.
    - Inertia values (I1, I2, I3) per config are shown as subtitle text.
    - Column title is 'Config N' not 'Batch N'.
    - m, l, g parameters are NOT needed (free rigid body, no pendulum physics).
    - I_diag is read from data['inertia_info'] stored by the datagen.
    """
    filename, file_path = _resolve_dataset_path(save_dir, filename)
    if file_path is None:
        return

    print(f"Loading: {file_path}")
    data = load_data(file_path)

    t = np.asarray(data['t'])              # (T,)
    # CHANGED: read inertia from the stored inertia_info list (one entry per config).
    # The pendulum used m, l, g kwargs; racket has no gravity and needs I_diag per config.
    inertia_info_list = data.get('inertia_info', [])

    train_data = data['x']
    test_data = data.get('test_x', None)
    num_configs = train_data.shape[0]  # CHANGED: num_us → num_configs

    # Global disturbance-torque range
    all_u = [np.linalg.norm(train_data[..., 12:15], axis=-1).flatten()]
    if test_data is not None:
        all_u.append(np.linalg.norm(test_data[..., 12:15], axis=-1).flatten())
    u_max = np.concatenate(all_u).max()
    norm = Normalize(vmin=0, vmax=max(u_max, 0.01))
    cmap = plt.get_cmap('viridis')

    nrows = 8
    ncols = num_configs
    fig = plt.figure(figsize=(4.5 * ncols, 3.5 * nrows))

    # Build the axes grid manually because rows 0 and 4 need 3D projection
    axes = [[None] * ncols for _ in range(nrows)]
    for r in range(nrows):
        for c in range(ncols):
            idx = r * ncols + c + 1
            if r in (0, 4):
                axes[r][c] = fig.add_subplot(nrows, ncols, idx, projection='3d')
            else:
                axes[r][c] = fig.add_subplot(nrows, ncols, idx)

    datasets = [
        ("Train", train_data, 0),
        ("Test",  test_data,  4),
    ]

    # Wireframe sphere coordinates (reused per panel)
    uu, vv = np.mgrid[0:2 * np.pi:24j, 0:np.pi:12j]
    sx = np.cos(uu) * np.sin(vv)
    sy_ = np.sin(uu) * np.sin(vv)
    sz = np.cos(vv)

    for ds_label, ds_data, row_offset in datasets:
        if ds_data is None:
            continue
        N_avail = ds_data.shape[2]
        n_plot = min(N_avail, num_trajs_to_plot)

        for cfg_idx in range(num_configs):  # CHANGED: u_idx → cfg_idx
            batch = ds_data[cfg_idx]              # (T, N, 15)
            R_all = batch[..., :9]                # (T, N, 9)
            omega_all = batch[..., 9:12]          # (T, N, 3)
            u_all = batch[..., 12:15]             # (T, N, 3)

            # CHANGED: extract e1 (long head axis, column 0 of R) instead of bob direction (column 2).
            # The pendulum tracked R @ e_z (bob direction = 3rd column).
            # For the racket we track R @ e1 (long axis = 1st column) to visualize orientation.
            e1_all = _rotmat_principal_axis(R_all, axis=0)           # (T, N, 3)

            omega_norm_all = np.linalg.norm(omega_all, axis=-1)      # (T, N)
            u_norm_all = np.linalg.norm(u_all, axis=-1)              # (T, N)

            # CHANGED: intermediate-axis alignment angle instead of tilt angle.
            # Pendulum used tilt α = arccos((R e_z)_z), a geometric angle of the pendulum.
            # For the racket we use α = arccos(|ω̂ · e₂|) to measure how aligned the spin
            # is with the unstable intermediate axis — the heart of the Dzhanibekov effect.
            e2_body = np.array([0.0, 1.0, 0.0])  # body-frame intermediate axis
            omega_hat = omega_all / (omega_norm_all[..., np.newaxis] + 1e-12)
            align_all = np.abs(omega_hat @ e2_body)                   # (T, N)
            align_angle_all = np.arccos(np.clip(align_all, 0.0, 1.0))  # α ∈ [0, π/2]

            # CHANGED: kinetic energy only — no potential energy (free rigid body, no gravity).
            # Pendulum used E = T + V with V = m g l (R e_z)_z.
            # Racket has no gravity, so T(t) = ½ ωᵀ I ω is the full mechanical energy.
            if cfg_idx < len(inertia_info_list):
                info = inertia_info_list[cfg_idx]
                I_diag = np.array([info['I1'], info['I2'], info['I3']])
            else:
                I_diag = np.ones(3)  # fallback
            # T = ½ (I1 ω1² + I2 ω2² + I3 ω3²)
            T_kin_all = 0.5 * np.sum(I_diag * omega_all**2, axis=-1)  # (T, N)

            dt = float(t[1] - t[0]) if len(t) > 1 else 1.0
            align_rate_all = np.gradient(align_angle_all, dt, axis=0)  # (T, N)

            # ── Row 0/4: e1 axis on S² ──
            # CHANGED: visualizes where the racket's long head axis (e1) points in world frame,
            # instead of the pendulum bob position. Reference markers are removed (no "up"/"down"
            # physical meaning for a free body) — replaced by a neutral equator ring.
            ax_s2 = axes[row_offset + 0][cfg_idx]
            ax_s2.plot_wireframe(sx, sy_, sz, color='lightgray',
                                 alpha=0.3, linewidth=0.5)
            # CHANGED: equator circle instead of north/south pole markers (no gravity reference)
            theta_eq = np.linspace(0, 2 * np.pi, 100)
            ax_s2.plot(np.cos(theta_eq), np.sin(theta_eq), np.zeros(100),
                       color='steelblue', linewidth=0.8, alpha=0.5)

            for trial in range(n_plot):
                e1 = e1_all[:, trial, :]
                u_n = u_norm_all[:, trial]
                pts = e1.reshape(-1, 1, 3)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                lc3d = Line3DCollection(segs, cmap=cmap, norm=norm,
                                        alpha=0.6, linewidth=1.2)
                lc3d.set_array(u_n[:-1])
                ax_s2.add_collection3d(lc3d)
                ax_s2.scatter(*e1[0], color='black', s=8, zorder=11)

            ax_s2.set_xlim(-1.1, 1.1)
            ax_s2.set_ylim(-1.1, 1.1)
            ax_s2.set_zlim(-1.1, 1.1)
            ax_s2.set_box_aspect([1, 1, 1])
            ax_s2.set_xlabel('x', fontsize=8)
            ax_s2.set_ylabel('y', fontsize=8)
            ax_s2.set_zlabel('z', fontsize=8)
            ax_s2.tick_params(labelsize=7)
            if row_offset == 0:
                # CHANGED: column title shows 'Config N' and inertia values
                if cfg_idx < len(inertia_info_list):
                    info = inertia_info_list[cfg_idx]
                    title_str = (f"Config {cfg_idx}\n"
                                 f"I1={info['I1']:.4f}  I2={info['I2']:.4f}  I3={info['I3']:.4f}")
                else:
                    title_str = f"Config {cfg_idx}"
                ax_s2.set_title(title_str, fontsize=9, fontweight='bold')
            if cfg_idx == 0:
                # CHANGED: row label updated from 'Bob on S²' to 'e1 axis on S²'
                ax_s2.text2D(-0.18, 0.5, f"{ds_label}\ne1 axis on S²",
                             transform=ax_s2.transAxes, fontsize=10,
                             rotation=90, va='center', ha='center', fontweight='bold')

            # ── Row 1/5: Intermediate-axis alignment phase portrait ──
            # CHANGED: x-axis is now α = arccos(|ω̂ · e₂|), the angle between ω
            # and the unstable axis e2. Pendulum used tilt α = arccos((R e_z)_z).
            # This directly visualizes the Dzhanibekov flip dynamics.
            ax_align = axes[row_offset + 1][cfg_idx]
            for trial in range(n_plot):
                a = align_angle_all[:, trial]
                ad = align_rate_all[:, trial]
                u_n = u_norm_all[:, trial]
                pts = np.array([a, ad]).T.reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                lc = LineCollection(segs, cmap=cmap, norm=norm,
                                    alpha=0.6, linewidth=1.0)
                lc.set_array(u_n[:-1])
                ax_align.add_collection(lc)
                ax_align.scatter(a[0], ad[0], color='black', s=8, zorder=3)
            ax_align.autoscale()
            ax_align.grid(True, alpha=0.3, linestyle='--')
            ax_align.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
            # CHANGED: axis label reflects intermediate-axis alignment, not tilt
            ax_align.set_xlabel(r"$\alpha = \arccos(|\hat{\omega}\cdot e_2|)$ (rad)", fontsize=9)
            if cfg_idx == 0:
                ax_align.set_ylabel(f"{ds_label}\n" + r"$\dot{\alpha}$ (rad/s)",
                                    fontsize=9)

            # ── Row 2/6: ||ω|| over time ──
            # (unchanged in structure; only label/variable name updates)
            ax_om = axes[row_offset + 2][cfg_idx]
            for trial in range(n_plot):
                om_n = omega_norm_all[:, trial]
                u_n = u_norm_all[:, trial]
                pts = np.array([t, om_n]).T.reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                lc = LineCollection(segs, cmap=cmap, norm=norm,
                                    alpha=0.6, linewidth=1.0)
                lc.set_array(u_n[:-1])
                ax_om.add_collection(lc)
            ax_om.autoscale()
            ax_om.grid(True, alpha=0.3, linestyle='--')
            ax_om.set_xlabel("t (s)", fontsize=9)
            if cfg_idx == 0:
                ax_om.set_ylabel(f"{ds_label}\n" + r"$\|\omega\|$ (rad/s)",
                                 fontsize=9)

            # ── Row 3/7: Kinetic energy over time ──
            # CHANGED: pendulum plotted total energy E = T + V (with gravity).
            # Racket is a free body: no gravity, so only kinetic energy T = ½ ωᵀ I ω.
            # For a torque-free trajectory this should be nearly constant (numerical check).
            ax_T = axes[row_offset + 3][cfg_idx]
            for trial in range(n_plot):
                T_k = T_kin_all[:, trial]
                u_n = u_norm_all[:, trial]
                pts = np.array([t, T_k]).T.reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                lc = LineCollection(segs, cmap=cmap, norm=norm,
                                    alpha=0.6, linewidth=1.0)
                lc.set_array(u_n[:-1])
                ax_T.add_collection(lc)
            ax_T.autoscale()
            ax_T.grid(True, alpha=0.3, linestyle='--')
            ax_T.set_xlabel("t (s)", fontsize=9)
            if cfg_idx == 0:
                # CHANGED: label updated from total energy E (J) to kinetic energy T (J)
                ax_T.set_ylabel(f"{ds_label}\n" + r"$T = \frac{1}{2}\omega^\top I\omega$ (J)",
                                fontsize=9)

    # Colorbar
    # CHANGED: label updated from wind force to disturbance torque
    cbar_ax = fig.add_axes([0.93, 0.12, 0.015, 0.76])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cb.set_label(r'$\|\tau_d\|$ (Nm)', fontsize=10)

    # CHANGED: figure title updated for the tennis racket / Dzhanibekov context
    fig.suptitle("Tennis Racket SO(3)-Aware Phase Space  "
                 r"(e2 = unstable intermediate axis, Dzhanibekov effect)",
                 fontsize=14, fontweight='bold', y=0.99)
    plt.subplots_adjust(right=0.91, hspace=0.4, wspace=0.3)

    ds_name = os.path.splitext(filename)[0]
    # CHANGED: output filename suffix retained as _phase_space_SO3
    out_path = os.path.join(save_dir, f'{ds_name}_phase_space_SO3.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


# ─────────────────── Entry point ───────────────────

if __name__ == "__main__":
    import argparse

    # CHANGED: description updated for tennis racket
    parser = argparse.ArgumentParser(description="Plot tennis racket phase space from a dataset .pkl file.")
    parser.add_argument("pkl_path", type=str, help="Path to the dataset .pkl file.")
    parser.add_argument("--num_trajs", type=int, default=15, help="Number of trajectories to plot per panel.")
    args = parser.parse_args()

    dataset_directory = os.path.dirname(os.path.abspath(args.pkl_path))
    filename = os.path.basename(args.pkl_path)

    # CHANGED: function names updated; m/l/g kwargs removed (not needed for free rigid body)
    plot_racket_phase_space(dataset_directory, num_trajs_to_plot=args.num_trajs, filename=filename)
    plot_so3_phase_space(dataset_directory, num_trajs_to_plot=args.num_trajs, filename=filename)
