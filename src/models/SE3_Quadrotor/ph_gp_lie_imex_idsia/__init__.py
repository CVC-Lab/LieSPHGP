"""PH-GP-LieIMEX, IDSIA variant: same model as ph_gp_lie_imex, with a robust likelihood,
horizon weighting and diverged-window masking in the training objective. All three are off by
default, so an unchanged config reproduces ph_gp_lie_imex exactly."""

MODEL_NAME = "ph_gp_lie_imex_idsia"
SOLVER = "lie-imex"

