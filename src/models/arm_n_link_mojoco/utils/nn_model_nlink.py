r"""MLP / PSD / MatrixNet subnetworks for the $n$-link arm — the **NN baseline**.

Self-contained port of ``src/utils/JAX/neural_networks.py`` (same architectures,
same orthogonal initialisation) so nothing outside ``src/models/arm_n_link/`` is
imported or modified.

These are the deterministic counterparts of the variational GPs in
``gp_model_nlink.py``. To keep the two families interchangeable inside
:mod:`ph_network_nlink`, every module here accepts the **same call signature**
as its GP twin — ``(x, key=None, inference_mode=False)`` — and simply ignores
the two extra arguments, since an MLP has no weight posterior to sample from.

    ==================  ===============  ==========================
    GP (variational)    NN (this file)   role
    ==================  ===============  ==========================
    ``GP_NLink``        ``MLP``          scalar / vector output
    ``PSD_GP_NLink``    ``PSD_NN``       PSD matrix ($M^{-1}$, $D$)
    ``MatrixGP_NLink``  ``MatrixNet_NN`` arbitrary matrix ($g$, $\Sigma$)
    ==================  ===============  ==========================

`weight_kl_loss()` returns exactly zero for every module here: the NN variants
are point estimates, so the ELBO's KL term is absent by construction rather
than switched off.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np


def choose_nonlinearity(name: str) -> Callable:
    fns = {'tanh': jnp.tanh, 'relu': jax.nn.relu, 'sigmoid': jax.nn.sigmoid,
           'softplus': jax.nn.softplus, 'selu': jax.nn.selu, 'elu': jax.nn.elu,
           'swish': jax.nn.swish}
    if name not in fns:
        raise ValueError(f'nonlinearity {name!r} not recognized')
    return fns[name]


def _orthogonal_linear(in_dim: int, out_dim: int, gain: float, *, key,
                       use_bias: bool = True, dtype=jnp.float32) -> eqx.nn.Linear:
    r"""`eqx.nn.Linear` with an orthogonally-initialised weight of scale `gain`.

    Mirrors ``torch.nn.init.orthogonal_(l.weight, gain=init_gain)``; Equinox
    defaults to a uniform init, so the weight is overwritten. `dtype` is pinned
    explicitly for the same reason as in the GP modules — the environment turns
    on ``jax_enable_x64`` at import, and an implicit default would silently give
    float64 parameters against a float32 batch.
    """
    k_layer, k_init = jax.random.split(key)
    layer = eqx.nn.Linear(in_dim, out_dim, use_bias=use_bias, key=k_layer)
    w = jax.nn.initializers.orthogonal(scale=gain)(k_init, (out_dim, in_dim),
                                                   dtype=dtype)
    layer = eqx.tree_at(lambda l: l.weight, layer, w)
    if use_bias:
        layer = eqx.tree_at(lambda l: l.bias, layer, layer.bias.astype(dtype))
    return layer


class MLP(eqx.Module):
    """3-layer MLP, tanh activations, no activation on the output layer."""
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    linear3: eqx.nn.Linear
    nonlinearity: Callable = eqx.field(static=True)

    def __init__(self, key, input_dim: int, hidden_dim: int, output_dim: int,
                 nonlinearity: str = 'tanh', bias_bool: bool = True,
                 init_gain: float = 1.0, dtype=jnp.float32):
        k1, k2, k3 = jax.random.split(key, 3)
        self.linear1 = _orthogonal_linear(input_dim, hidden_dim, init_gain,
                                          key=k1, dtype=dtype)
        self.linear2 = _orthogonal_linear(hidden_dim, hidden_dim, init_gain,
                                          key=k2, dtype=dtype)
        self.linear3 = _orthogonal_linear(hidden_dim, output_dim, init_gain,
                                          key=k3, use_bias=bias_bool, dtype=dtype)
        self.nonlinearity = choose_nonlinearity(nonlinearity)

    def __call__(self, x, key: Optional[jax.Array] = None,
                 inference_mode: bool = False):
        del key, inference_mode          # signature parity with the GP modules
        h = self.nonlinearity(self.linear1(x))
        h = self.nonlinearity(self.linear2(h))
        return self.linear3(h)

    def weight_kl_loss(self):
        """Zero — a point estimate has no variational posterior."""
        return jnp.zeros(())


class PSD_NN(eqx.Module):
    r"""MLP whose output is a positive-semidefinite $d\times d$ matrix.

    The net emits the $\tfrac{d(d+1)}{2}$ entries of a lower-triangular $L$ and
    returns $LL^\top\succeq0$. Following the reference implementation the
    diagonal is offset by $\sqrt{\varepsilon}$ (no softplus, no tanh cap) — so
    unlike :class:`PSD_GP_NLink` the diagonal is unbounded above and may pass
    through zero. That is deliberate: it is the baseline's parameterisation, and
    keeping it faithful is what makes the GP-vs-NN comparison meaningful.
    """
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    linear3: eqx.nn.Linear
    linear4: eqx.nn.Linear
    nonlinearity: Callable = eqx.field(static=True)
    diag_dim:     int   = eqx.field(static=True)
    off_diag_dim: int   = eqx.field(static=True)
    epsilon:      float = eqx.field(static=True)
    _tril_rows: Tuple[int, ...] = eqx.field(static=True)
    _tril_cols: Tuple[int, ...] = eqx.field(static=True)

    def __init__(self, key, input_dim: int, hidden_dim: int, diag_dim: int,
                 nonlinearity: str = 'tanh', init_gain: float = 0.5,
                 epsilon: float = 0.0, dtype=jnp.float32):
        assert diag_dim > 1
        self.diag_dim = int(diag_dim)
        self.off_diag_dim = self.diag_dim * (self.diag_dim - 1) // 2
        self.epsilon = float(epsilon)

        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.linear1 = _orthogonal_linear(input_dim, hidden_dim, init_gain, key=k1, dtype=dtype)
        self.linear2 = _orthogonal_linear(hidden_dim, hidden_dim, init_gain, key=k2, dtype=dtype)
        self.linear3 = _orthogonal_linear(hidden_dim, hidden_dim, init_gain, key=k3, dtype=dtype)
        self.linear4 = _orthogonal_linear(hidden_dim,
                                          self.diag_dim + self.off_diag_dim,
                                          init_gain, key=k4, dtype=dtype)
        self.nonlinearity = choose_nonlinearity(nonlinearity)

        rows, cols = np.tril_indices(self.diag_dim, k=-1)
        self._tril_rows = tuple(int(r) for r in rows)
        self._tril_cols = tuple(int(c) for c in cols)

    def __call__(self, q, key=None, inference_mode=False):
        del key, inference_mode
        h = self.nonlinearity(self.linear1(q))
        h = self.nonlinearity(self.linear2(h))
        h = self.nonlinearity(self.linear3(h))
        out = self.linear4(h)

        diag = out[:self.diag_dim] + jnp.sqrt(self.epsilon)
        L = jnp.diag(diag)
        L = L.at[jnp.array(self._tril_rows), jnp.array(self._tril_cols)].set(
            out[self.diag_dim:])
        return L @ L.T

    def weight_kl_loss(self):
        return jnp.zeros(())


class MatrixNet_NN(eqx.Module):
    """MLP whose flat output is reshaped to a matrix of the given `shape`."""
    mlp: MLP
    shape: Tuple[int, int] = eqx.field(static=True)

    def __init__(self, key, input_dim: int, hidden_dim: int, shape: tuple,
                 nonlinearity: str = 'tanh', init_gain: float = 1.0,
                 dtype=jnp.float32):
        self.shape = tuple(shape)
        self.mlp = MLP(key, input_dim, hidden_dim,
                       int(shape[0]) * int(shape[1]),
                       nonlinearity=nonlinearity, init_gain=init_gain, dtype=dtype)

    def __call__(self, x, key=None, inference_mode=False):
        del key, inference_mode
        return self.mlp(x).reshape(self.shape)

    def weight_kl_loss(self):
        return jnp.zeros(())
