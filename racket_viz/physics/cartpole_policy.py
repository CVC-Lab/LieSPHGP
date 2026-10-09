"""Hand-written forward pass for the coworker's real trained ct_sac actor --
NOT a from-scratch derivation like every other physics module in this
project, since there's no equation to derive: the "physics" here is
literally 229,500 numbers in `data/cartpole_policy.json`
(`ct_sac_cartpole_top_300000_steps.pth`, exported by the coworker's own
`export_policy.py`) plus a fixed architecture, both specified in full in
`POLICY_SPEC.md` (not committed to this repo -- it's the coworker's
document, cited here for provenance) and cross-checked live against the
actual weights file before being trusted (see DECISIONS.md, 2026-08-12):
24/24 of the coworker's own PyTorch-verified test vectors matched to within
2.13e-6 N against a stated 1e-4 N tolerance.

Architecture (fixed, not configurable -- this is what the checkpoint IS):
    input 48
      body.0   Linear(48  -> 400)   ReLU
      body.2   Linear(400 -> 300)   ReLU
      body.4   Linear(300 -> 300)   NO activation  <-- the one real gotcha;
                                                        create_mlp only puts
                                                        activations BETWEEN
                                                        hidden dims, so the
                                                        last body layer is
                                                        bare linear
      mu       Linear(300 -> 1)     no activation
      tanh, then affine rescale to [action_low, action_high]
    output 1   (force on the cart, newtons)

Observation: 16 frames of (cos theta, sin theta, x), oldest-first, 48 raw
floats, NO normalization of any kind (confirmed by the coworker: no
VecNormalize, no running mean/std anywhere on the obs->network path). On
reset, all 16 frames are filled with the INITIAL measurement (not zeros),
matching the training env's own reset behavior -- an all-identical window
encodes zero velocity information, by construction, not a bug.

This policy was trained at dt=0.01 (the 16-frame window spans 0.16s of
history) and has only ever seen one plant (mp=0.1, mc=1.0, l=0.5 half-
length, g=9.8, force limit +-10N, wind sigma_gust=0.002, sigma_turb=0.001) --
both facts matter for how a caller drives this module, not for the module
itself, which is just the forward pass and window bookkeeping.
"""
import json

import numpy as np


def load_policy(path):
    """Load the exported policy JSON (see cartpole_policy.json's own
    "format": "ct_sac_actor_v1" schema -- {obs, layers: [{W,b,activation}],
    output: {rescale: {low, high}}}).

    @param path str or Path
    @returns dict, the parsed JSON
    """
    with open(path) as f:
        return json.load(f)


def forward(obs, policy):
    """The exact forward pass: alternating Linear+activation body layers,
    a bare-linear `mu` head, then tanh-squash and affine rescale to the
    action box.

    @param obs (48,) array-like, raw (not normalized) observation
    @param policy dict, as returned by load_policy
    @returns float, force in Newtons
    """
    h = np.asarray(obs, dtype=np.float64)
    for layer in policy["layers"]:
        W = np.asarray(layer["W"], dtype=np.float64)
        b = np.asarray(layer["b"], dtype=np.float64)
        h = W @ h + b
        if layer["activation"] == "relu":
            h = np.maximum(h, 0.0)
    low = policy["output"]["rescale"]["low"]
    high = policy["output"]["rescale"]["high"]
    z = np.tanh(h)
    action = low + (high - low) * (z + 1.0) / 2.0
    return float(action[0])


def init_window(theta, x):
    """The observation window at reset: all 16 frames filled with the SAME
    initial measurement (not zeros) -- matches the training env's own
    reset, and means the implied velocity starts at exactly zero.

    @param theta float, pole angle (radians, 0 = upright)
    @param x float, cart position (m)
    @returns list of 16 [cos_theta, sin_theta, x] frames, oldest-first
    """
    frame = [np.cos(theta), np.sin(theta), x]
    return [list(frame) for _ in range(16)]


def push_frame(window, theta, x):
    """Advances the window by one step: drop the oldest frame, append the
    new measurement as the newest. Does not mutate `window`.

    @param window list of 16 [cos_theta, sin_theta, x] frames, oldest-first
    @param theta float @param x float
    @returns a NEW list of 16 frames
    """
    return window[1:] + [[np.cos(theta), np.sin(theta), x]]


def window_to_observation(window):
    """Flattens the 16x3 window into the 48-float vector `forward` expects,
    in the same oldest-first order the window is already kept in.

    @param window list of 16 [cos_theta, sin_theta, x] frames
    @returns list of 48 floats
    """
    return [v for frame in window for v in frame]
