"""Lie-PH models on SE(3) x R^6 (quadrotor, simulated PyBullet and real IDSIA flights): port-Hamiltonian SDE / ODE with
GP or MLP subnetworks, Lie-IMEX integrator, EKF marginal likelihood (or the trajectory loss for Lie-PH-NN-ODE).
Same algorithm as src/models/3D_SO3_Windy_Pendulum/lie_ph. See README.md.
"""
