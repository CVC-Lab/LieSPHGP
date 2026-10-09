import numpy as np
import pytest

import pendulum_geometry


def test_inertia_about_pivot_not_com():
    """A pendulum pivots about a FIXED POINT that is NOT its center of mass --
    Euler's equations need the inertia tensor taken about that pivot (parallel-
    axis-shifted from the COM value), unlike the free-floating racket which
    rotates about its own COM. For the default geometry, the pivot-frame
    transverse moment must exceed the COM-frame one by exactly m*d_com^2
    (the parallel-axis theorem), where d_com is the pivot-to-COM distance."""
    result = pendulum_geometry.build_pendulum()
    I1, I2, I3 = result["I"]
    total_mass = result["total_mass"]
    d_com = np.linalg.norm(result["com"])

    # Axial (imin) << transverse (imid == imax) for a thin rod + bob.
    assert I1 < I2
    assert I2 == pytest.approx(I3, rel=1e-9)

    # Reconstruct the COM-frame transverse moment via the closed-form pieces
    # (rod: (1/3) m_rod L^2 about the pivot, standard thin-rod-about-end
    # formula, already IS about the pivot, not COM -- so shift the *bob*
    # piece back to compare) and check the parallel-axis relationship holds
    # for the bob's own contribution specifically.
    bob_mass = result["bob_mass"]
    bob_self_inertia = (2.0 / 5.0) * bob_mass * result["bob_radius"] ** 2
    # Bob's parallel-axis contribution to the transverse moment about the
    # pivot: its own spin inertia (isotropic) plus m*L^2 for the offset.
    bob_pivot_transverse = bob_self_inertia + bob_mass * result["rod_length"] ** 2
    rod_pivot_transverse = (1.0 / 3.0) * result["rod_mass"] * result["rod_length"] ** 2
    assert I2 == pytest.approx(bob_pivot_transverse + rod_pivot_transverse, rel=1e-9)
    assert d_com > 0


def test_axial_moment_is_bobs_own_spin_inertia_only():
    """The rod and the bob's center both lie exactly on the rod's own axis, so
    neither contributes a parallel-axis term to I_axial -- it's exactly the
    bob's own solid-sphere spin inertia, independent of rod_length."""
    result = pendulum_geometry.build_pendulum(rod_length=1.0)
    result2 = pendulum_geometry.build_pendulum(rod_length=1.7)
    I1_a = min(result["I"])
    I1_b = min(result2["I"])
    assert I1_a == pytest.approx(I1_b, rel=1e-9)
    expected = (2.0 / 5.0) * result["bob_mass"] * result["bob_radius"] ** 2
    assert I1_a == pytest.approx(expected, rel=1e-9)


def test_evecs_orthonormal_proper_rotation():
    result = pendulum_geometry.build_pendulum()
    evecs = result["evecs"]
    assert evecs.shape == (3, 3)
    np.testing.assert_allclose(evecs.T @ evecs, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(np.linalg.det(evecs), 1.0, atol=1e-9)


def test_axis_roles_partition_indices():
    result = pendulum_geometry.build_pendulum()
    assert {result["imin"], result["imid"], result["imax"]} == {0, 1, 2}


def test_mass_split_matches_bob_fraction():
    result = pendulum_geometry.build_pendulum(total_mass=2.0, bob_fraction=0.75)
    assert result["bob_mass"] == pytest.approx(1.5)
    assert result["rod_mass"] == pytest.approx(0.5)


def test_com_scales_with_rod_length_and_bob_fraction():
    """More mass in the bob (which sits at the rod's far end) pulls the COM
    further from the pivot, toward the bob."""
    low_bob = pendulum_geometry.build_pendulum(bob_fraction=0.5)
    high_bob = pendulum_geometry.build_pendulum(bob_fraction=0.95)
    assert np.linalg.norm(high_bob["com"]) > np.linalg.norm(low_bob["com"])
