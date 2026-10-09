"""Rod + spherical-bob pendulum geometry -> inertia tensor about the PIVOT.

Unlike the free-floating tennis racket (which rotates about its own center of
mass), a pendulum rotates about a FIXED PIVOT point that is generally NOT its
center of mass -- Euler's equations (`rigid_body.py`) require the inertia
tensor taken about that actual center of rotation, so this computes I about
the pivot (the origin, before the rod+bob's own diagonalization) directly,
not about the COM with a parallel-axis correction bolted on afterward.

Geometry: a thin rod of length `rod_length` from the pivot (origin) to a
solid spherical bob of radius `bob_radius` at its far end, both lying along
one axis. Closed-form (no point discretization needed, unlike
racket_geometry.py's numerical quadrature -- both pieces have exact standard
formulas):
  - thin rod about one end:      I_transverse_rod = (1/3) m_rod L^2,  I_axial_rod = 0
  - solid sphere about its own center (isotropic): I_self = (2/5) m_bob r^2
  - bob shifted to the pivot via the parallel-axis theorem:
      I_axial_bob      = I_self                      (bob's center lies ON the axis)
      I_transverse_bob = I_self + m_bob * L^2

I_axial ends up small (order bob_radius^2) and I_transverse >> I_axial for any
physically reasonable rod_length/bob_radius ratio -- spin about the rod's own
axis is nearly free, swinging about either transverse axis is not. The two
transverse axes are exactly equal (this body is rotationally symmetric about
the rod), unlike the racket's 3 generically-distinct moments -- a real
"spherical pendulum" property, not a modeling shortcut.
"""
import numpy as np


def build_pendulum(
    rod_length=1.0,
    bob_radius=0.15,
    total_mass=1.0,
    bob_fraction=0.80,
):
    """Build pendulum geometry and its principal-axis inertia tensor about the pivot.

    Returns a dict with:
      I                 : (I1, I2, I3), I1 <= I2 <= I3 (ascending, matches
        racket_geometry.py's convention)
      evecs             : (3, 3) columns are principal axes in the pre-rotation
        (rod-along-local-x) frame
      imin, imid, imax  : int, indices 0/1/2 -- imin is always the (near-free)
        axial spin axis; imid/imax are the two (equal) transverse swing axes
      com               : (3,) center of mass in the principal-axis (body) frame,
        needed for computing gravity torque (which acts at the COM, not the pivot)
      rod_length, bob_radius, rod_mass, bob_mass, total_mass : geometry/mass
        parameters, passed through for rendering and torque calculations
    """
    rod_mass = (1.0 - bob_fraction) * total_mass
    bob_mass = bob_fraction * total_mass

    # Pre-rotation frame: rod from the pivot (origin) along local x to the bob
    # center at (rod_length, 0, 0). Both pieces already lie exactly on this
    # axis, so I is already diagonal here -- eigh below just sorts/labels it.
    I_axial = (2.0 / 5.0) * bob_mass * bob_radius ** 2
    I_transverse = (
        (1.0 / 3.0) * rod_mass * rod_length ** 2
        + bob_mass * (rod_length ** 2 + (2.0 / 5.0) * bob_radius ** 2)
    )
    I_tensor = np.diag([I_axial, I_transverse, I_transverse])

    com_pre = np.array([
        (rod_mass * (rod_length / 2.0) + bob_mass * rod_length) / total_mass,
        0.0,
        0.0,
    ])

    evals, evecs = np.linalg.eigh(I_tensor)
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]
    if np.linalg.det(evecs) < 0:
        evecs[:, -1] *= -1

    com_body = com_pre @ evecs

    return {
        "I": evals,
        "evecs": evecs,
        "imin": 0,
        "imid": 1,
        "imax": 2,
        "com": com_body,
        "rod_length": rod_length,
        "bob_radius": bob_radius,
        "rod_mass": rod_mass,
        "bob_mass": bob_mass,
        "total_mass": total_mass,
    }
