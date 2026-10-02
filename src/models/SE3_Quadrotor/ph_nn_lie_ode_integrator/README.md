# PH-NN Lie ODE integrator (JAX)

This independent package contains its JAX network, losses, data loading,
checkpointing, training loop, and second-order Lie-Heun integrator. Rotation is
advanced with the exponential map, so the numerical state remains on the
rotation group.

`network.py` exposes one `DissipativeSE3HamNODE` model with the six physical
subnetworks. Loss equations and Lie integration remain in their own files.
