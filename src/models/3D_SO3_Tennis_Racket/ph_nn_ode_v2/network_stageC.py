"""Stage C network: SO(3) Hamiltonian NODE with inertia-augmented state.

Extends network.py by adding `inertia_dim` extra inputs to all sub-networks.
The state vector is (vec(R)∈ℝ⁹, I₁,I₂,I₃∈ℝ³, ω∈ℝ³, u∈ℝ³) = ℝ¹⁸.

Sub-networks receive q_ext = (vec(R), I₁,I₂,I₃) ∈ ℝ¹² so they can learn
the inertia dependence: M_net learns diag(1/I₁,1/I₂,1/I₃) directly from
the embedded inertia values.  This breaks the Stage A/B single-config
restriction and enables multi-config generalization.

Differences from network.py:
  - `inertia_dim` constructor parameter (default 3); rotmatdim = 9 + inertia_dim.
  - forward() splits x as (q_ext, q_dot, u) with rotmatdim=12.
  - Cross-product geometry uses q_ext[:, :9] (the rotation subspace only).
  - dHdq is 12D; only dHdq[:, :9] enters the Lie bracket.
  - JVP tangent is padded: (dR, 0₃) since d(inertia)/dt = 0.
  - Return is 18D: cat(dq₉, zeros₃, ddq₃, zeros₃).
  - FixedInertiaFromState reads I₁,I₂,I₃ from q_ext[:, 9:12] directly.
"""
import torch
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../utils')))
from ode_nn_models import MLP, PSD, MatrixNet


class FixedInertiaFromState(torch.nn.Module):
    """Drop-in M_net for Stage C: reads I₁,I₂,I₃ from q_ext[:, 9:12].

    The state is (vec(R), I₁,I₂,I₃, ω, u), so each sample carries its own
    inertia.  This module returns diag(1/I₁, 1/I₂, 1/I₃) per sample without
    any learnable parameters — use it to pin M_net to ground truth and check
    that V/Dw/g learn correctly before releasing M_net.
    """
    def forward(self, q_ext):
        I_vals = q_ext[:, 9:12]                        # (N, 3)
        return torch.diag_embed(1.0 / I_vals)          # (N, 3, 3)


class DissipativeSO3HamNODE(torch.nn.Module):
    def __init__(self, M_net=None, Dw_net=None, V_net=None, g_net=None,
                 device=None, u_dim=3, init_gain=0.01, friction=True,
                 inertia_dim=3):
        super().__init__()
        self.inertia_dim = inertia_dim            # extra dims appended after vec(R)
        self.rotmatdim   = 9 + inertia_dim        # input dim to all sub-nets
        self.angveldim   = 3
        self.friction    = friction

        # epsilon=1.0 (fp32 stability fix)
        self.M_net = M_net or PSD(self.rotmatdim, 20, self.angveldim,
                                  init_gain=init_gain, epsilon=1.0).to(device)

        if friction:
            self.Dw_net = Dw_net or PSD(self.rotmatdim, 20, self.angveldim,
                                        init_gain=init_gain, epsilon=0.0).to(device)
        self.V_net = V_net or MLP(self.rotmatdim, 20, 1, init_gain=init_gain).to(device)

        self.u_dim = u_dim
        if g_net is None:
            if u_dim == 1:
                self.g_net = MLP(self.rotmatdim, 20, self.angveldim).to(device)
            else:
                self.g_net = MatrixNet(self.rotmatdim, 20, self.angveldim * self.u_dim,
                                       shape=(self.angveldim, self.u_dim),
                                       init_gain=init_gain).to(device)
        else:
            self.g_net = g_net

        self.device = device
        self.nfe = 0

    def forward(self, t, x):
        with torch.enable_grad():
            self.nfe += 1
            bs = x.shape[0]
            zero_vec      = torch.zeros(bs, self.u_dim,      dtype=x.dtype, device=x.device)
            zero_inertia  = torch.zeros(bs, self.inertia_dim, dtype=x.dtype, device=x.device)

            if not x.requires_grad:
                x = x.detach().requires_grad_(True)

            # x = (q_ext[rotmatdim], q_dot[angveldim], u[u_dim])
            #   = (vec(R)[9], I₁I₂I₃[3], ω[3], u[3]) for inertia_dim=3
            q_ext, q_dot, u = torch.split(
                x, [self.rotmatdim, self.angveldim, self.u_dim], dim=1)

            # Rotation sub-block of q_ext — used for all geometry (cross products).
            R_flat = q_ext[:, :9]       # (B, 9)

            # ── Two-call cat-split structure (same as base network) ──
            M_q = self.M_net(q_ext)
            q_dot_aug = torch.unsqueeze(q_dot, dim=2)
            p = torch.squeeze(torch.linalg.solve(M_q, q_dot_aug), dim=2)

            q_p = torch.cat((q_ext, p), dim=1)
            q_ext_split, p = torch.split(q_p, [self.rotmatdim, self.angveldim], dim=1)
            R_flat_split = q_ext_split[:, :9]   # rotation part after cat-split

            M_q_inv = self.M_net(q_ext_split)
            V_q     = self.V_net(q_ext_split)
            g_q     = self.g_net(q_ext_split)
            Dw_q    = self.Dw_net(q_ext_split)

            p_aug = torch.unsqueeze(p, dim=2)
            H = (torch.squeeze(torch.matmul(torch.transpose(p_aug, 1, 2),
                                             torch.matmul(M_q_inv, p_aug))) / 2.0
                 + torch.squeeze(V_q))

            dH = torch.autograd.grad(H.sum(), q_p, create_graph=True)[0]
            # dH has shape (B, rotmatdim + angveldim); first rotmatdim entries are
            # dH/d(q_ext) = [dH/d(vec R), dH/d(I₁), dH/d(I₂), dH/d(I₃)].
            # Only the rotation sub-block enters the Lie bracket.
            dHdq_ext, dHdp = torch.split(dH, [self.rotmatdim, self.angveldim], dim=1)
            dHdR = dHdq_ext[:, :9]     # (B, 9) — gradient w.r.t. vec(R) only

            if self.u_dim == 1:
                F = g_q * u
            else:
                F = torch.squeeze(torch.matmul(g_q, torch.unsqueeze(u, dim=2)))

            # ── #2: Batched cross products ──
            q_3x3    = R_flat_split.view(-1, 3, 3)               # (B, 3, 3)
            dHdp_b   = dHdp.unsqueeze(1).expand(-1, 3, -1)       # (B, 3, 3)
            dq = torch.linalg.cross(q_3x3, dHdp_b, dim=2).reshape(-1, 9)

            dHdR_3x3 = dHdR.view(-1, 3, 3)                       # (B, 3, 3)
            grav = torch.linalg.cross(q_3x3, dHdR_3x3, dim=2).sum(dim=1)  # (B, 3)

            if self.friction:
                dp = (torch.linalg.cross(p, dHdp, dim=1)
                      + grav
                      - torch.squeeze(torch.matmul(Dw_q, torch.unsqueeze(dHdp, dim=2)))
                      + F)
            else:
                dp = torch.linalg.cross(p, dHdp, dim=1) + grav + F

            # ── #1: dM_inv/dt via JVP ──
            # Inertia is constant: d(I₁,I₂,I₃)/dt = 0.  Pad dq to rotmatdim with zeros.
            dq_ext_dt = torch.cat(
                [dq, torch.zeros(bs, self.inertia_dim, dtype=x.dtype, device=x.device)],
                dim=1)
            _, dM_inv_dt = torch.func.jvp(self.M_net, (q_ext_split,), (dq_ext_dt,))

            ddq = (torch.squeeze(torch.matmul(M_q_inv, torch.unsqueeze(dp, dim=2)), dim=2)
                   + torch.squeeze(torch.matmul(dM_inv_dt, torch.unsqueeze(p, dim=2)), dim=2))

            # Return state derivative: (dR[9], d_inertia=0[3], dω[3], du=0[3])
            return torch.cat((dq, zero_inertia, ddq, zero_vec), dim=1)
