User: ok there are 4 stages we can tweak for v2 1st dataset level like maybe add more data 2nd optimizer level maybe change adam optimizer or tweak learning rate or do mini batch wise sampling 3rd algorithm the physics equations / math level and loss function 4th gaussian process level now for each stage suggest me changes that can make our gp-imex v2 version better/best for higher obs noise like 0.05, 0.1 0.25 .. (if you feel I missed out any stages then you can come up with new stages i just gave you an idea of how stages look like) dont make any changes lets discuss first


Claude: Discussion only, no code touched. I checked the V2 loop once more so every suggestion points at a real line. First the frame, then the stages, then a priority order.

**The one fact that decides what matters at high noise.** With $K=10$ steps at $h=0.01$ the window is $0.1$ s. The signal in a window grows with time, the noise does not:

| channel | signal over 0.1 s (aggressive data) | σ = 0.05 | σ = 0.1 | σ = 0.25 |
|---|---|---|---|---|
| position, $\tfrac12 a t^2$ with $a\approx 2$ m/s² | 0.01 m | below noise | below noise | far below |
| rotation, $\|\omega\| t$ with $\omega\approx 3$ rad/s | 0.3 rad | ok | marginal | below noise |
| $\omega$, $\|\Delta\omega\|$ | 2.9 rad/s | ok | ok | ok |
| noise floor $\sum_c 3\sigma_c^2$ | | 0.03 | 0.12 | 0.75 |

Two consequences. First, at σ = 0.1 and above the model learns almost only from the velocity channels, so anything that makes the velocity channels cleaner or the window longer helps more than anything else. Second, the measured failure is a bias, not variance: the power-balance test gave learned dissipation 1.6× too high at σ = 0.05 and 4× at σ = 0.1. That is the errors-in-variables effect. The rollout starts from a noisy $\tilde x_0=x_0+\epsilon$, and the cheapest way for the model to explain "the noisy start does not match the noisy future" is to damp harder. More data cannot remove a bias, because averaging shrinks variance by $1/N$ and leaves the bias where it is. Every stage below is judged by whether it removes that bias, raises window SNR, or only lowers variance.

**Stage 1. Dataset.**
1. More flights. Cheap, and worth a 144 → 576 check, but expect a plateau for the reason above. It reduces variance in $g$, $M_2^{-1}$ and lengthscales, not the over-damping.
2. Longer windows. Position signal grows as $t^2$, rotation as $t$, noise stays flat, so $K=20$ or $30$ roughly doubles or triples window SNR for free. V2 hard-codes non-overlapping $K=10$ (train.py:81, run_noise_sweep.py:78). Train with $K=10$ then $K=20$ as a curriculum so the early rollouts do not blow up.
3. Overlapping windows, stride 1 instead of 10. Ten times more windows from the same flights. They are correlated, so the count $N$ in $\beta/N$ should stay the number of independent flights, not windows.
4. Use the aggressive dataset from tmp. It raised per-window $|\Delta\omega|$ from 0.38 to 2.9 rad/s and torque std by 4×. The velocity channels are the only ones that survive at σ = 0.1, so this is the single biggest dataset lever.
5. Realistic per-channel noise for the σ = 0.25 study. The same absolute σ on all 18 channels means rotation-matrix entries get σ = 0.25 on values in $[-1,1]$. That is not a noise level any sensor has and the noisy $R$ is no longer a rotation, so the Lie-IMEX step gets a non-orthogonal initial state. Either project $\tilde R_0$ to $SO(3)$ by polar decomposition, or set the rotation noise as a random rotation $\tilde R=R\exp(\hat\eta)$, $\eta\sim\mathcal N(0,\sigma_R^2 I)$. Without this, σ = 0.25 measures the projection error, not the model.
6. Fresh noise draw per epoch. Legitimate only as a diagnostic, since we generate the noise. If the model still over-damps with fresh noise every epoch, the bias is in the estimator, not in fitting one noise realisation.

**Stage 2. Optimizer.** This stage is the weakest lever. Adam is per-parameter scale invariant, so "the loss is small" is never the reason weights do not move. What matters is the gradient ratio between data term and KL, which is Stage 3.
1. Minibatches of 128 to 256 windows. Same steps cost 5 to 10× less, so 50k steps become affordable, and the gradient noise mildly regularises. Nothing more.
2. Cosine decay $10^{-3}\to10^{-5}$ over the run. The clean control plateaued at $2\times10^{-3}$ partly because the step never shrank.
3. Two learning rates. Lengthscales and log-stds at $10^{-4}$, weight means at $10^{-3}$. A lengthscale that collapses is the way a GP overfits noise, and it collapses fastest when it shares the mean's learning rate.
4. Checkpoint selection on a validation split of flights, never on test. Both the NN and GP reports currently pick the best step on test data.
5. Optional finish with L-BFGS on the posterior mean only, with sampling off, since the objective is then deterministic. Cheap and often gives another 20 to 30 % on the clean control.

**Stage 3. Physics and loss.** This is where the noise robustness is won or lost.
1. Latent initial states. Make $x_0^{(n)}$ a free variable per window with prior $\mathcal N(\tilde x_0^{(n)},\sigma^2 I)$, on rotations $R_0=\tilde R_0\exp(\hat\eta_0)$. The objective becomes
$$\mathcal L=\sum_n\Big[\sum_{k=1}^{K}\frac{\|\Phi_k(x_0^{(n)},w)-\tilde x_k^{(n)}\|^2}{2\sigma^2}+\frac{\|x_0^{(n)}-\tilde x_0^{(n)}\|^2}{2\sigma^2}\Big]+\beta\,\mathrm{KL}(q(w)\,\|\,p(w)).$$
Each window then has 11 observations constraining 18 latent numbers, so it is well posed, and the over-damping escape route disappears because the model may move the start instead of damping the future. This is the change that directly targets the measured 1.6×/4× dissipation bias. Cost is $N\times18$ extra parameters, about 26k for 1440 windows.
2. Proper ELBO instead of trajectory MSE. V2 currently uses `--data-fit trajectory-mse` with $\beta=10^{-2}$ (run_noise_sweep.py:22, :80), which makes the data-to-KL ratio depend on σ by accident. Use $\tfrac{1}{2\sigma^2}\|e\|^2$ with the known σ, scale 1, and $\beta$ chosen on validation once. Then the same $\beta$ is correct at every noise level, and the posterior stds stop drifting upward as they did in the V2 reports.
3. Per-channel-group σ. Even with the same absolute noise, position, rotation entries, $v$ and $ω$ enter the dynamics with different gains. Four learnable σ values, one per group, with a prior centred on the known σ, cost four parameters and stop the rotation entries from dominating the residual at high noise.
4. Structure that fixes the gauge. From the substitution test the rotational operators co-adapt: replacing $M_2^{-1}$ alone made things worse, replacing the whole block gave $10^{-2}$. So carry over the V3 items: position-only $V$ with a linear $z$ feature so gravity extrapolates, a scale matrix $S$ on $g$ so the unobservable off-block entries stay near zero, and a tighter dissipation prior, $D=D_{\text{nominal}}(1+0.25\,r_\theta)$ instead of a free multiplier. At high noise, everything the data cannot identify must come from the prior, so a loose prior on $D$ is exactly what the over-damping bias exploits.
5. Energy balance stays a metric, not a loss. At σ = 0.05 and 0.1 the residual over $\Delta H$ was 32 % and 127 %, so a loss on it would only train the model to fit noise. Keep it as a report row.

**Stage 4. Gaussian process.**
1. Antithetic weight sampling. `sample_weights` draws one $w=\mu+\sigma_w\epsilon$ per step (model.py:171–184). Use the pair $\mu\pm\sigma_w\epsilon$ and average the two losses. The first-order effect of the draw cancels, so $\sigma_w$ can be allowed to grow again without the $J^{-1}\approx4\times10^{4}$ amplification wrecking the rollout. That is how the model stops being MAP with $\log\sigma_w=-5$ (model.py:154) and becomes a GP whose posterior means something.
2. Prior scales in the units the dynamics amplify. Choose each residual scale $s_k$ so that a unit prior weight produces an acceleration that is a fixed fraction of the data's, $s_k\,\|\mathcal M^{-1}\|\,\|\phi\|\approx0.1\,\|\dot\xi\|_{\text{data}}$. Right now the scales are set in operator units, which is why a $10^{-3}$ residual on $V$ was ten times the control torque.
3. Lengthscale priors and floors. Put a log-normal prior on each lengthscale and log the lengthscales per checkpoint. If they shrink as σ grows, the GP is overfitting through the kernel, and a floor at roughly the data's spacing is the fix. This is the GP-native regulariser, and no one has looked at it yet.
4. More features only after 3. Going from 128 to 512 features at the current settings buys capacity the data cannot identify. With lengthscale priors in place, 256 to 512 is safe and will help $g(x,R)$ most.
5. Low-rank plus diagonal posterior. Mean-field cannot represent the correlated directions the substitution test exposed, between $M_2^{-1}$, $g_\omega$ and $D_\omega$, so it under-reports uncertainty exactly where it is largest. A rank-8 correction per subnetwork is small and gives honest bands for the calibration figure.
6. Not worth it. Sparse inducing-point GPs give nothing over random features at this size, and deep kernels remove the interpretability that is the paper's argument.

**Two stages you did not list.**
- Stage 0, evaluation. Decide what "better at σ = 0.25" means before running anything. The noisy test loss has a floor of 0.75, so a win must be measured on the clean test targets that are already stored in the pickle, on operator errors, and on the closed-loop position error. Otherwise every model ties at the floor.
- Stage 5, observation model and pre-processing. Projection of $\tilde R_0$ onto $SO(3)$, whether the controller sees raw or filtered states, and the substep count of the integrator. Small, but at σ = 0.25 they are the difference between a meaningful run and garbage.

**Priority order for 0.05 and 0.1, with what I expect.**

| order | change | stage | expected effect |
|---|---|---|---|
| 1 | latent initial states | 3.1 | removes most of the 1.6×/4× dissipation bias |
| 2 | ELBO with known σ, scale 1, $\beta$ from validation | 3.2 | stable posterior stds, same setting at all noise levels |
| 3 | gauge structure: position-only $V$, $S$ on $g$, tight $D$ prior | 3.4 | fixes D0 extrapolation and control failure |
| 4 | aggressive data plus $K=20$ curriculum | 1.2, 1.4 | 2 to 4× window SNR |
| 5 | antithetic sampling plus lengthscale priors | 4.1, 4.3 | uncertainty becomes real, no kernel overfitting |
| 6 | minibatch, cosine decay, validation checkpoints | 2 | cheaper runs, honest model selection |

For σ = 0.25, items 1, 3 and the rotation-noise fix in Stage 1.5 are prerequisites, and I would frame the result as operator identification and closed-loop tracking rather than rollout accuracy, because the rollout metric is at the floor regardless of model.

If you agree, the natural first run is items 1 and 2 together on the existing V2 data at σ = 0.1, with the power-balance dissipation ratio as the gate. If the ratio drops from 4× toward 1.2×, the diagnosis is confirmed and the rest is polishing. Tell me which items to put into the V3 change list and whether V3 lives beside V2 or under tmp, and I will write the list out before touching anything.