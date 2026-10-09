"""PH-GP-SDE on SE(3) x R^6 for the BlueROV2: Lie-IMEX SDE integrator + EKF marginal likelihood.

Copy of src/models/SE3_Quadrotor/lie_ph (same algorithm, same model variants) with n_u inputs, an optional
full SPD translational inverse mass (model.M1_form) and an optional rotation term in the potential (model.V_rotation).
"""
