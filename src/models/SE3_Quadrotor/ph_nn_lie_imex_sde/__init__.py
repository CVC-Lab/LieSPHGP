"""PH-NN-LieIMEX-SDE: the ph_nn_lie_imex model with additive process noise on the body twist.

Same six MLP subnetworks and the same Lie-IMEX integrator, plus two learned process-noise scales and four learned
observation scales. Training rolls S sample paths per window and scores them with a moment-matched Gaussian
likelihood, so the sample spread has to match the residual spread and the process scales receive a data-fit
signal. With ``sde.enabled = false`` the model is the same likelihood with a single deterministic path (no
process noise), which is the matched control for the ODE/SDE comparison.

Note this package's objective is the moment-matched NLL, not the ``trajectory-mse`` of ph_nn_lie_imex: a
Gaussian likelihood is what makes a noise scale identifiable at all.
"""

MODEL_NAME = "ph_nn_lie_imex_sde"
SOLVER = "lie-imex"
