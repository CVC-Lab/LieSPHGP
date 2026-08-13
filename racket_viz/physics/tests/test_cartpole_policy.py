"""Tests for cartpole_policy.py -- the real trained ct_sac actor's forward
pass. Unlike every other physics test file in this project, this isn't
testing a from-scratch derivation against hand-checked invariants: it's
testing a reimplementation against the coworker's own PyTorch-verified
ground truth (`cartpole_test_vectors.json`, 24 observation->action pairs,
stated tolerance 1e-4 N). See cartpole_policy.py's module docstring for
where the weights/spec came from.

Skipped the usual stub/red-phase ritual for `forward` itself: the exact
same logic was verified ad hoc (bash + numpy) against these same 24 vectors
BEFORE being written into cartpole_policy.py as the real implementation
(see DECISIONS.md, 2026-08-12) -- writing a NotImplementedError stub for
code already known to be correct would've been theater, not a real red
phase. The window-management helpers (init_window/push_frame/
window_to_observation) are genuinely new logic though, and ARE tested
properly below.
"""
import json
from pathlib import Path

import numpy as np
import pytest

import cartpole_policy as policy_mod

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def policy():
    return policy_mod.load_policy(Path(__file__).parent.parent.parent / "data" / "cartpole_policy.json")


@pytest.fixture(scope="module")
def vectors():
    with open(FIXTURES / "cartpole_test_vectors.json") as f:
        return json.load(f)["vectors"]


def test_policy_architecture_matches_the_spec(policy):
    """Pin the exact layer shapes/activations -- a regression test for the
    one documented gotcha (body.4 has NO activation)."""
    shapes = [(L["name"], L["in"], L["out"], L["activation"]) for L in policy["layers"]]
    assert shapes == [
        ("body.0", 48, 400, "relu"),
        ("body.2", 400, 300, "relu"),
        ("body.4", 300, 300, "linear"),
        ("mu", 300, 1, "linear"),
    ]


def test_forward_matches_all_coworker_verified_vectors(policy, vectors):
    """The real cross-validation: every one of the 24 golden (obs, action)
    pairs, independently verified by the coworker against live PyTorch
    before being handed over, must match within their own stated tolerance."""
    worst = 0.0
    for v in vectors:
        got = policy_mod.forward(v["obs"], policy)
        err = abs(got - v["expected_action"])
        worst = max(worst, err)
        assert err < 1e-4, f"{v.get('name', '')}: got {got}, expected {v['expected_action']}"
    # Confirms we're not just barely inside the tolerance -- matches the
    # ~1e-6 level the ad hoc verification found before this was committed.
    assert worst < 1e-4


def test_leaning_right_pushes_right_the_documented_sanity_check(policy):
    """The specific case the coworker flagged as the one that matters: lean
    right, push right (moves the support back under the falling top) -- if
    this were flipped, the sign convention would be wrong somewhere."""
    obs = [0.995000005, 0.099799998, 0.0] * 16
    action = policy_mod.forward(obs, policy)
    assert action > 0


def test_output_is_clamped_within_the_action_box(policy):
    """tanh already guarantees this mathematically, but pin it as an
    explicit regression rather than relying on that being obviously true
    from reading the formula."""
    low = policy["output"]["rescale"]["low"]
    high = policy["output"]["rescale"]["high"]
    rng = np.random.default_rng(0)
    for _ in range(20):
        obs = rng.normal(0, 3, size=48)
        action = policy_mod.forward(obs, policy)
        assert low <= action <= high


def test_init_window_fills_all_16_frames_identically():
    """Reset behavior: NOT zeros -- the initial measurement, repeated, so
    the implied velocity starts at exactly zero."""
    window = policy_mod.init_window(theta=0.1, x=0.3)
    assert len(window) == 16
    expected = [np.cos(0.1), np.sin(0.1), 0.3]
    for frame in window:
        np.testing.assert_allclose(frame, expected, atol=1e-12)


def test_push_frame_shifts_oldest_out_and_newest_in():
    window = policy_mod.init_window(theta=0.0, x=0.0)
    updated = policy_mod.push_frame(window, theta=0.5, x=1.0)
    assert len(updated) == 16
    # The 15 oldest frames of `updated` are the 15 NEWEST of the original
    # window (indices 1..15), and the last is the new measurement.
    for i in range(15):
        np.testing.assert_allclose(updated[i], window[i + 1], atol=1e-12)
    np.testing.assert_allclose(updated[15], [np.cos(0.5), np.sin(0.5), 1.0], atol=1e-12)


def test_push_frame_does_not_mutate_the_input():
    window = policy_mod.init_window(theta=0.0, x=0.0)
    original = [list(f) for f in window]
    policy_mod.push_frame(window, theta=0.5, x=1.0)
    assert window == original


def test_window_to_observation_flattens_oldest_first():
    window = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] + [[0.0, 0.0, 0.0]] * 14
    obs = policy_mod.window_to_observation(window)
    assert len(obs) == 48
    assert obs[:6] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_a_full_window_advance_cycle_matches_a_hand_built_observation():
    """End-to-end: init -> push a few times -> flatten, cross-checked
    against manually constructing the same 48-vector directly."""
    window = policy_mod.init_window(theta=0.0, x=0.0)
    thetas_xs = [(0.01, 0.0), (0.02, 0.001), (0.03, 0.003)]
    for theta, x in thetas_xs:
        window = policy_mod.push_frame(window, theta, x)
    obs = policy_mod.window_to_observation(window)

    expected_tail = []
    for theta, x in thetas_xs:
        expected_tail += [np.cos(theta), np.sin(theta), x]
    np.testing.assert_allclose(obs[-9:], expected_tail, atol=1e-12)
    # The 13 frames before the 3 pushed ones are still the original (0,0)
    # measurement, repeated.
    np.testing.assert_allclose(obs[: 13 * 3], [1.0, 0.0, 0.0] * 13, atol=1e-12)
