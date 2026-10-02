"""PH-GP-LieIMEX-SDE, IDSIA variant: the ph_gp_lie_imex_idsia model with additive process noise on the
momenta (forces and torques), integrated by a stochastic Lie-IMEX step, and trained with a moment-matched
predictive likelihood over S sample paths. With sde.enabled = false it is ph_gp_lie_imex_idsia exactly."""

MODEL_NAME = "ph_gp_lie_imex_sde_idsia"
SOLVER = "lie-imex"
