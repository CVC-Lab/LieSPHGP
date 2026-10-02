● User: thoughts on the three V2 reports?

  Claude: They confirm the diagnosis from the training curves, and they add one new
  failure that matters for the paper: the zero-mean $V(x,R)$ does not transfer to
  the D0 evaluation data. Numbers first.

  Clean test-window loss at step 10000, training protocol

  ┌───────┬─────────────────────┬─────────────────────┬─────────────────────┐
  │ model │     $\sigma=0$      │    $\sigma=0.05$    │    $\sigma=0.1$     │
  ├───────┼─────────────────────┼─────────────────────┼─────────────────────┤
  │ V1    │ $4.2\times10^{-3}$  │ $5.3\times10^{-3}$  │ $1.19\times10^{-2}$ │
  ├───────┼─────────────────────┼─────────────────────┼─────────────────────┤
  │ V2    │ $2.24\times10^{-2}$ │ $2.08\times10^{-2}$ │ $2.05\times10^{-2}$ │
  └───────┴─────────────────────┴─────────────────────┴─────────────────────┘

  V2 ends at the same value for all three noise levels. That is the signature of an
  error that has nothing to do with noise: the model is still slowly unlearning the
  spurious torque from the random $R$-dependent potential it started with, loss 29
  at step 0, and at step 10000 it has not finished. Five times worse than V1 on
  clean data, and still falling at 40 percent per thousand steps, so the number is a
  snapshot of an unconverged run rather than a property of the architecture.

  Table 1, 0.5 s rollouts on the D0 flights

  ┌───────────────────┬────────────────────┬──────────┬───────────────────────┐
  │                   │      V1 clean      │ V2 clean │    V2 $\sigma=0.1$    │
  ├───────────────────┼────────────────────┼──────────┼───────────────────────┤
  │ $v^2$ mean        │ $5.4\times10^{-3}$ │ $3.54$   │ $0.86$                │
  ├───────────────────┼────────────────────┼──────────┼───────────────────────┤
  │ $x^2$ mean        │ $1.4\times10^{-4}$ │ $0.105$  │ $0.033$               │
  ├───────────────────┼────────────────────┼──────────┼───────────────────────┤
  │ $\omega^2$ mean   │ $0.134$            │ $3.26$   │ $0.83$                │
  ├───────────────────┼────────────────────┼──────────┼───────────────────────┤
  │ $|\Delta E|$ mean │ $4\times10^{-4}$   │ $0.106$  │ $0.046$               │
  ├───────────────────┼────────────────────┼──────────┼───────────────────────┤
  │ controller RMSE   │ 4.5 cm             │ 28 cm    │ 3.4 m, 58% saturation │
  └───────────────────┴────────────────────┴──────────┴───────────────────────┘

  A velocity error of 1.9 m/s RMS in half a second is a 40 percent gravity error.
  This is the zero-mean stationary GP doing what it always does outside its training
  support: the D0 flights visit positions the aggressive set barely covers, and a
  Matérn potential with no mean reverts toward zero there, so gravity fades. V1's
  $mgz$ mean, or a linear $z$ feature, is exactly what prevents this. Dropping the
  mean for fairness is fine; dropping it without a linear term is not.

  What V2 got right. Isotropic $M_1$ works. Gauge-fixed $M_1^{-1}$ error is
  $3\times10^{-5}$ against $7\times10^{-3}$ for V1, two hundred times better, and
  $D_v$, $D_\omega$ are near exact on clean data. The raw $M_1^{-1}$ and $V$ errors
  are large, 5 to 25 and 0.107, but that is the expected free overall scale once the
  gravity mean is gone, the same gauge freedom the NN has, and the report's gauge
  fix removes it.

  What still needs fixing, in order of damage
  1. $V$ input back to $x$ only, with a linear $z$ feature so gravity is
     representable and extrapolates. This removes both the slow-torque problem and
     the D0 transfer failure.
  2. The $g$ mask. Control degrades with noise exactly as before, 28 cm to 1.9 m to
     3.4 m, and gauge-fixed $g$ error grows with noise, $7\times10^{-3}$ to
     $9.6\times10^{-3}$.
  3. The objective. Posterior stds are 0.35 for $M_1$, 0.68 for $D_v$, 0.47 for $g$
     under MSE with $\beta=0.01$, the KL-imbalance again. NLL with fixed $\sigma$
     and scale 1, or $\beta$ divided by $2\sigma^2$.

  One thing to keep in mind when reading V2 against V1: V1 at $\sigma=0$ and V2 at
  $\sigma=0$ differ by three design changes at once, so the comparison cannot
  attribute the difference to any one of them. The gate-based sequence I proposed
  avoids that: V3 with the three fixes, clean run to step 3000, pass if window loss
  is below $2\times10^{-4}$, and only then the noisy pair. I can write V3 as a
  package next to V2 or under tmp/, whichever you prefer. Confirm and I will list
  the edits first.