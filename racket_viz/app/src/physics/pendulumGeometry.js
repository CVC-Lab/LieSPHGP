/**
 * Rod + spherical-bob pendulum geometry -> inertia tensor about the PIVOT.
 * Ported from physics/pendulum_geometry.py -- see that file's docstring for
 * the closed-form derivation (thin rod about one end + solid sphere shifted
 * to the pivot via the parallel-axis theorem; no point discretization needed,
 * unlike racketGeometry.js's numerical quadrature).
 *
 * Unlike the free-floating racket (inertia about its own COM), a pendulum
 * pivots about a FIXED POINT that is generally NOT its COM -- this computes
 * I about that pivot directly, which is what Euler's equations (rigidBody.js)
 * need for a body rotating about a fixed point other than its own COM.
 */
import { eigh3x3 } from "./eigen3x3.js";

/**
 * @param {object} [params]
 * @param {number} [params.rodLength=1.0]
 * @param {number} [params.bobRadius=0.15]
 * @param {number} [params.totalMass=1.0]
 * @param {number} [params.bobFraction=0.80]
 * @returns {{
 *   I: number[], evecs: number[][], imin: number, imid: number, imax: number,
 *   com: number[], rodLength: number, bobRadius: number,
 *   rodMass: number, bobMass: number, totalMass: number
 * }}
 */
export function buildPendulum(params = {}) {
  const { rodLength = 1.0, bobRadius = 0.15, totalMass = 1.0, bobFraction = 0.8 } = params;

  const rodMass = (1.0 - bobFraction) * totalMass;
  const bobMass = bobFraction * totalMass;

  const Iaxial = (2.0 / 5.0) * bobMass * bobRadius ** 2;
  const Itransverse =
    (1.0 / 3.0) * rodMass * rodLength ** 2 + bobMass * (rodLength ** 2 + (2.0 / 5.0) * bobRadius ** 2);

  const Itensor = [
    [Iaxial, 0, 0],
    [0, Itransverse, 0],
    [0, 0, Itransverse],
  ];

  const comPre = [(rodMass * (rodLength / 2.0) + bobMass * rodLength) / totalMass, 0, 0];

  const { values: I, vectors: evecs } = eigh3x3(Itensor);

  const com = [
    comPre[0] * evecs[0][0] + comPre[1] * evecs[1][0] + comPre[2] * evecs[2][0],
    comPre[0] * evecs[0][1] + comPre[1] * evecs[1][1] + comPre[2] * evecs[2][1],
    comPre[0] * evecs[0][2] + comPre[1] * evecs[1][2] + comPre[2] * evecs[2][2],
  ];

  return {
    I,
    evecs,
    imin: 0,
    imid: 1,
    imax: 2,
    com,
    rodLength,
    bobRadius,
    rodMass,
    bobMass,
    totalMass,
  };
}
