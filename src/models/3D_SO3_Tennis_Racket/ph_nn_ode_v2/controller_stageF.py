"""Stage F: Geometric Attitude Controller on SO(3) × ℝ³.

Extends Stage D (proportional ω-only controller) to stabilise a full
target (R*, ω*) using the geodesic attitude error on SO(3).

Control law
-----------
    u  =  −K_R · e_R  −  K_p · e_ω

    e_R   = vee( logm(R*ᵀ R) )    ∈ ℝ³   geodesic attitude error
    e_ω   = ω − ω*                ∈ ℝ³   angular velocity error

IDA-PBC interpretation
----------------------
This is IDA-PBC with desired Hamiltonian

    H_d(R, ω) = H(R, ω) + ½ K_R ‖e_R‖² + ½ K_p ‖e_ω‖²

and J_d = J (interconnection unchanged), R_d = R + K_p I (added damping).

Linearised closed-loop per axis i (second-order oscillator):
    ωn_i = √(K_R / Iᵢ)
    ζ_i  = K_p / (2 √(K_R · Iᵢ))

Stability notes
---------------
  • Almost-global: stable for any initial (R₀, ω₀) except the antipodal
    set {θ_err = π}, which has measure zero on SO(3).
  • Same ZOH bound as Stage D: K_p < 2·I₁/dt  (set by the smallest inertia).
  • K_R has a much softer ZOH bound ≈ 4·I₁/dt² and is not a practical concern.

When K_R = 0 this reduces exactly to the Stage D proportional controller.
"""
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# SO(3) geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def hat(v):
    """Skew-symmetric matrix (hat map) for v ∈ ℝ³.

    hat(v) ω  =  v × ω   for any ω ∈ ℝ³.
    """
    v = np.asarray(v, dtype=np.float64).ravel()
    return np.array([[  0.0, -v[2],  v[1]],
                     [ v[2],   0.0, -v[0]],
                     [-v[1],  v[0],   0.0]])


def vee(Omega):
    """Axial vector of a skew-symmetric matrix (inverse of hat).

    vee(hat(v)) = v  for any v ∈ ℝ³.
    """
    return np.array([Omega[2, 1], Omega[0, 2], Omega[1, 0]])


def logm_SO3(R):
    """Matrix logarithm of R ∈ SO(3).

    Returns the unique skew-symmetric Ω ∈ so(3) with ‖vee(Ω)‖ ≤ π
    such that expm(Ω) = R.  Uses the Rodrigues formula:

        Ω = θ / (2 sin θ) · (R − Rᵀ),    θ = arccos((tr R − 1)/2)

    Special cases:
      θ ≈ 0 (near identity):  first-order approximation (R−Rᵀ)/2.
      θ ≈ π (antipodal):       reconstruct axis from symmetric part of R.
    """
    cos_theta = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))

    if theta < 1e-7:
        return (R - R.T) / 2.0

    if abs(theta - np.pi) < 1e-4:
        # R = 2 n nᵀ − I  →  (R + I)/2 = n nᵀ
        sym = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(sym)))
        n = sym[:, i] / np.sqrt(max(sym[i, i], 1e-15))
        return theta * hat(n)

    return (theta / (2.0 * np.sin(theta))) * (R - R.T)


def expm_SO3(Omega):
    """Matrix exponential of Ω ∈ so(3) (Rodrigues formula).

    Useful for constructing target rotations, e.g.:
        R_target = expm_SO3(hat([0, np.pi/4, 0]))   # 45° about e₂
    """
    v = vee(Omega)
    theta = float(np.linalg.norm(v))
    if theta < 1e-7:
        return np.eye(3) + Omega
    n = v / theta
    return (np.cos(theta) * np.eye(3)
            + np.sin(theta) * hat(n)
            + (1.0 - np.cos(theta)) * np.outer(n, n))


# ─────────────────────────────────────────────────────────────────────────────
# Controller
# ─────────────────────────────────────────────────────────────────────────────

class GeometricAttitudeController:
    """Full-state stabiliser for (R*, ω*) ∈ SO(3) × ℝ³.

    Parameters
    ----------
    R_star     : (3,3) target orientation in SO(3).
    omega_star : (3,)  target angular velocity in the body frame [rad/s].
    K_R        : orientation gain [N·m/rad].  Zero → Stage D proportional ctrl.
    K_p        : angular velocity gain [N·m·s/rad].
    clip       : torque saturation [N·m].
    """

    def __init__(self, R_star, omega_star, K_R=0.10, K_p=0.10, clip=2.0):
        self.R_star     = np.asarray(R_star,     dtype=np.float64).reshape(3, 3)
        self.omega_star = np.asarray(omega_star, dtype=np.float64).reshape(3)
        self.K_R  = float(K_R)
        self.K_p  = float(K_p)
        self.clip = float(clip)

    def attitude_error(self, R):
        """e_R = vee(logm(R*ᵀ R)) ∈ ℝ³  (zero when R = R*)."""
        return vee(logm_SO3(self.R_star.T @ np.asarray(R).reshape(3, 3)))

    def __call__(self, R_flat, omega):
        """Compute control torque u = −K_R e_R − K_p e_ω, clipped to ±clip."""
        R     = np.asarray(R_flat, dtype=np.float64).reshape(3, 3)
        omega = np.asarray(omega,  dtype=np.float64).ravel()
        e_R   = self.attitude_error(R)
        e_w   = omega - self.omega_star
        u = -self.K_R * e_R - self.K_p * e_w
        return np.clip(u, -self.clip, self.clip)

    def errors(self, R_flat, omega):
        """Returns (‖e_R‖ [rad], ‖e_ω‖ [rad/s]).

        Both are zero at the target (R*, ω*).
        ‖e_R‖ lies in [0, π];  ‖e_R‖ = π is the worst-case antipodal point.
        """
        R     = np.asarray(R_flat, dtype=np.float64).reshape(3, 3)
        omega = np.asarray(omega,  dtype=np.float64).ravel()
        e_R   = self.attitude_error(R)
        e_w   = omega - self.omega_star
        return float(np.linalg.norm(e_R)), float(np.linalg.norm(e_w))
