"""Per-subnetwork physics-target MSE diagnostics for the tennis-racket SO(3) NODE.

Copied from subnet_diagnostics.py and adapted for the free rigid body.

Key differences from the pendulum version:
  - M_tgt  = diag(1/I1, 1/I2, 1/I3)   (M_net outputs M⁻¹; here that is I⁻¹)
  - V_tgt  = 0                           (free body, no gravitational potential)
  - Dw_tgt = 0                           (Stage A: torque-free data, no friction)
  - g_tgt  = I₃                          (direct body-frame torque input, same as pendulum)

Note on the pendulum version's M_tgt:
  subnet_diagnostics.py uses M_tgt = (m·l²)·I₃ = M (the mass matrix), but M_net
  outputs M⁻¹ = (1/m·l²)·I₃.  For the default m=l=1 these are both I₃ so the
  error is invisible.  Here we correctly compare M_pred against M⁻¹.

Dw_tgt = 0 is appropriate for Stage A (torque-free dataset).  When friction data
is added (Stage B), pass friction_coeff and set Dw_tgt = friction_coeff · I₃.
"""
import torch


def _unwrap(model):
    return model._orig_mod if hasattr(model, '_orig_mod') else model


@torch.no_grad()
def subnet_physics_mse_tennis(
    model,
    x_hat,
    *,
    I1: float,
    I2: float,
    I3: float,
    friction_coeff: float = 0.0,
):
    """Compute mean MSE between each subnetwork's outputs and the true physics.

    Args:
        model:           DissipativeSO3HamNODE (or torch.compile wrapper).
        x_hat:           odeint output, shape (T, B, 15) — initial frame dropped.
        I1, I2, I3:      principal moments of inertia [kg·m²], I1 < I2 < I3.
        friction_coeff:  scalar viscous friction (0 for Stage A torque-free data).

    Returns:
        dict with scalar floats: {'M_loss', 'V_loss', 'Dw_loss', 'g_loss'}.
    """
    inner = _unwrap(model)
    device = x_hat.device
    dtype = x_hat.dtype

    # Use only predicted timesteps (drop the ground-truth initial condition)
    pred = x_hat[1:]                                 # (T-1, B, 15)
    T1, B, _ = pred.shape
    flat = pred.reshape(T1 * B, 15)
    q = flat[:, :9]                                  # (N, 9)
    N = q.shape[0]

    # ── Subnet outputs ──────────────────────────────────────────────────
    M_pred  = inner.M_net(q)                         # (N, 3, 3)  — this is M⁻¹
    V_pred  = inner.V_net(q).squeeze(-1)             # (N,)
    Dw_pred = inner.Dw_net(q)                        # (N, 3, 3)
    g_pred  = inner.g_net(q)                         # (N, 3, 3)

    # ── Ground-truth targets ─────────────────────────────────────────────
    I3_eye = torch.eye(3, device=device, dtype=dtype)

    # M⁻¹ = diag(1/I1, 1/I2, 1/I3) — constant, independent of R
    I_inv_diag = torch.tensor([1.0/I1, 1.0/I2, 1.0/I3], device=device, dtype=dtype)
    M_tgt = torch.diag(I_inv_diag).unsqueeze(0).expand(N, 3, 3)

    # V = 0 (free body; center both before MSE to be gauge-invariant)
    V_pred_c = V_pred - V_pred.mean()
    V_tgt_c  = torch.zeros_like(V_pred_c)

    # Dw = friction_coeff · I₃ (zero for Stage A, non-zero once friction enabled)
    Dw_tgt = (friction_coeff * I3_eye).unsqueeze(0).expand(N, 3, 3)

    # g = I₃ (direct body-frame torque input)
    g_tgt = I3_eye.unsqueeze(0).expand(N, 3, 3)

    # ── Mean MSE per subnet ──────────────────────────────────────────────
    M_loss  = (M_pred  - M_tgt ).pow(2).mean().item()
    V_loss  = (V_pred_c - V_tgt_c).pow(2).mean().item()
    Dw_loss = (Dw_pred - Dw_tgt).pow(2).mean().item()
    g_loss  = (g_pred  - g_tgt ).pow(2).mean().item()

    return {'M_loss': M_loss, 'V_loss': V_loss,
            'Dw_loss': Dw_loss, 'g_loss': g_loss}
