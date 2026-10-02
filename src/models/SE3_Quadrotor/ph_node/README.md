# PH-NODE (JAX)

This package trains the neural port-Hamiltonian vector field entirely in JAX
and advances its coordinate state with classical fourth-order Runge--Kutta.
The only command-line argument is the required timestamped YAML config.

The package is independent. `network.py` exposes one
`DissipativeSE3HamNODE` model with six named physical subnetworks,
`losses.py` defines the objective, `integrator.py` defines RK4, and `train.py`
owns the training loop.
