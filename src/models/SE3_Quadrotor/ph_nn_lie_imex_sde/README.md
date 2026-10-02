# PH-NN Lie-IMEX (JAX)

This independent package contains its JAX network, losses, data loading,
checkpointing, training loop, and second-order Lie-IMEX integrator. The
dissipative velocity blocks are solved implicitly and rotation is updated
through the exponential map.

`network.py` exposes one `DissipativeSE3HamNODE` model with the six physical
subnetworks. Loss equations and Lie-IMEX integration remain in their own files.
