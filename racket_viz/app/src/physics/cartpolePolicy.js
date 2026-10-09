/**
 * Hand-written forward pass for the coworker's real trained ct_sac actor.
 * Ported from physics/cartpole_policy.py -- see that module's docstring
 * for provenance (POLICY_SPEC.md, export_policy.py) and for why this one
 * skipped the usual "derive the physics from scratch" framing: there's no
 * equation here, just 229,500 numbers (data/cartpole_policy.json) and a
 * fixed architecture, cross-checked against the coworker's own
 * PyTorch-verified test vectors before being trusted.
 *
 * Loading the weights themselves (fetch("/cartpole_policy.json") in the
 * browser) is the caller's job, same as any other static asset -- this
 * module only ever takes an already-parsed policy object.
 */

/**
 * @param {number[]} obs (48,) raw (not normalized) observation
 * @param {object} policy parsed cartpole_policy.json
 * @returns {number} force in Newtons
 */
export function forward(obs, policy) {
  let h = obs;
  for (const layer of policy.layers) {
    const { W, b, activation } = layer;
    const next = new Array(W.length);
    for (let i = 0; i < W.length; i++) {
      let sum = b[i];
      const row = W[i];
      for (let j = 0; j < row.length; j++) sum += row[j] * h[j];
      next[i] = activation === "relu" ? Math.max(sum, 0.0) : sum;
    }
    h = next;
  }
  const { low, high } = policy.output.rescale;
  const z = Math.tanh(h[0]);
  return low + (high - low) * (z + 1.0) / 2.0;
}

/**
 * The observation window at reset: all 16 frames filled with the SAME
 * initial measurement (not zeros) -- matches the training env's own
 * reset, so the implied velocity starts at exactly zero.
 * @param {number} theta pole angle (radians, 0 = upright)
 * @param {number} x cart position (m)
 * @returns {number[][]} 16 [cos_theta, sin_theta, x] frames, oldest-first
 */
export function initWindow(theta, x) {
  const frame = [Math.cos(theta), Math.sin(theta), x];
  return Array.from({ length: 16 }, () => [...frame]);
}

/**
 * Advances the window by one step: drop the oldest frame, append the new
 * measurement as the newest. Does not mutate `window`.
 * @param {number[][]} window 16 frames, oldest-first
 * @param {number} theta @param {number} x
 * @returns {number[][]} a NEW array of 16 frames
 */
export function pushFrame(window, theta, x) {
  return [...window.slice(1), [Math.cos(theta), Math.sin(theta), x]];
}

/**
 * Flattens the 16x3 window into the 48-float vector `forward` expects, in
 * the same oldest-first order the window is already kept in.
 * @param {number[][]} window
 * @returns {number[]}
 */
export function windowToObservation(window) {
  return window.flat();
}
