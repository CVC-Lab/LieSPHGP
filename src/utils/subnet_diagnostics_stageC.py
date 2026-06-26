"""Per-subnetwork physics-target MSE diagnostics for Stage C (18D state).

Adapted from subnet_diagnostics_tennis.py for the inertia-augmented state:
  state = (vec(R)[9], I₁I₂I₃[3], ω[3], u[3]) = ℝ¹⁸.

Sub-networks receive q_ext = (vec(R), I₁,I₂,I₃) ∈ ℝ¹², so M_tgt is per-sample:
  M_tgt[i] = diag(1/I₁[i], 1/I₂[i], 1/I₃[i])
where I₁,I₂,I₃ are read from q_ext[:, 9:12].

Key differences from subnet_diagnostics_tennis.py:
  - No I1,I2,I3 kwargs — inertia is embedded in the state.
  - flat = pred.reshape(T1 * B, 18) (18D, not 15D).
  - q_ext = flat[:, :12] passed to sub-networks.
  - M_tgt is per-sample (N, 3, 3), not broadcast from scalars.
"""
import torch


def _unwrap(model):
    return model._orig_mod if hasattr(model, '_orig_mod') else model


@torch.no_grad()
def subnet_physics_mse_stageC(
    model,
    x_hat,
    *,
    friction_coeff: float = 0.0,
):
    """Compute mean MSE between each subnetwork's outputs and the true physics.

    Args:
        model:           DissipativeSO3HamNODE with inertia_dim=3.
        x_hat:           odeint output, shape (T, B, 18) — initial frame included.
        friction_coeff:  scalar viscous friction (0 for Stage C torque-free data).

    Returns:
        dict with scalar floats: {'M_loss', 'V_loss', 'Dw_loss', 'g_loss'}.
    """
    inner = _unwrap(model)
    device = x_hat.device
    dtype = x_hat.dtype

    # Drop the initial condition (ground-truth) timestep
    pred = x_hat[1:]                                 # (T-1, B, 18)
    T1, B, _ = pred.shape
    flat = pred.reshape(T1 * B, 18)
    q_ext = flat[:, :12]                             # (N, 12) = vec(R) + I₁I₂I₃
    N = q_ext.shape[0]

    # ── Subnet outputs ──────────────────────────────────────────────────
    M_pred  = inner.M_net(q_ext)                     # (N, 3, 3) — this is M⁻¹
    V_pred  = inner.V_net(q_ext).squeeze(-1)         # (N,)
    Dw_pred = inner.Dw_net(q_ext)                    # (N, 3, 3)
    g_pred  = inner.g_net(q_ext)                     # (N, 3, 3)

    # ── Ground-truth targets ─────────────────────────────────────────────
    I3_eye = torch.eye(3, device=device, dtype=dtype)

    # M⁻¹ = diag(1/I₁, 1/I₂, 1/I₃) per sample, read from embedded inertia
    I_vals = q_ext[:, 9:12]                          # (N, 3)
    M_tgt = torch.diag_embed(1.0 / I_vals)           # (N, 3, 3)

    # V = 0 (free body; centre before MSE to be gauge-invariant)
    V_pred_c = V_pred - V_pred.mean()
    V_tgt_c  = torch.zeros_like(V_pred_c)

    # Dw = friction_coeff · I₃ (zero for Stage C torque-free data)
    Dw_tgt = (friction_coeff * I3_eye).unsqueeze(0).expand(N, 3, 3)

    # g = I₃ (direct body-frame torque input)
    g_tgt = I3_eye.unsqueeze(0).expand(N, 3, 3)

    # ── Mean MSE per subnet ──────────────────────────────────────────────
    M_loss  = (M_pred  - M_tgt  ).pow(2).mean().item()
    V_loss  = (V_pred_c - V_tgt_c).pow(2).mean().item()
    Dw_loss = (Dw_pred  - Dw_tgt ).pow(2).mean().item()
    g_loss  = (g_pred   - g_tgt  ).pow(2).mean().item()

    return {'M_loss': M_loss, 'V_loss': V_loss,
            'Dw_loss': Dw_loss, 'g_loss': g_loss}
