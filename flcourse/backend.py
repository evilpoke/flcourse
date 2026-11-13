"""A tiny NumPy / PyTorch compatibility layer.

Every aggregator, pre-aggregator, attack and DP mechanism in ``flcourse`` accepts

* a 2-D ``numpy.ndarray`` or ``torch.Tensor`` of shape ``(n, d)`` (one row per client), or
* a ``list`` / ``tuple`` of 1-D numpy arrays or 1-D torch tensors,

and returns its result in the *same library* (for torch: on the same device).
This is the same contract as ByzFL (EPFL).

Inside an aggregator we write::

    X, xp = as_matrix(vectors)   # X: (n, d) matrix, xp: NUMPY or TORCH ops
    return xp.median(X, axis=0)

``xp`` is a small, NumPy-flavoured namespace whose functions behave *identically*
for both libraries. Two classic pitfalls are handled here once and for all:

* ``median``: ``torch.median`` returns the *lower* middle value for an even number of
  rows, ``numpy.median`` the *average* of both. ``xp.median`` always averages.
* ``std``: ``torch.std`` is unbiased by default, ``numpy.std`` is not. ``xp.std`` always
  uses the sample standard deviation (``ddof=1``) like ByzFL's ALIE attack.
"""
from __future__ import annotations

from typing import Any, Sequence, Tuple

import numpy as np
import torch

Array = Any  # numpy.ndarray or torch.Tensor


def is_torch(x: Any) -> bool:
    return isinstance(x, torch.Tensor)


# ----------------------------------------------------------------------------------
# NumPy implementation
# ----------------------------------------------------------------------------------
class _NumpyOps:
    name = "numpy"

    # reductions ---------------------------------------------------------------------
    @staticmethod
    def mean(X, axis=0):
        return X.mean(axis=axis)

    @staticmethod
    def sum(X, axis=0):
        return X.sum(axis=axis)

    @staticmethod
    def std(X, axis=0):
        return X.std(axis=axis, ddof=1)

    @staticmethod
    def sort(X, axis=0):
        return np.sort(X, axis=axis)

    @staticmethod
    def argsort(X, axis=-1):
        return np.argsort(X, axis=axis, kind="stable")

    @staticmethod
    def median(X, axis=0):
        return _median_from_sorted(np.sort(X, axis=axis), axis)

    @staticmethod
    def norm(X, axis=-1):
        """Euclidean (l2) norm along ``axis``; for a 1-D vector returns a 0-d array."""
        return np.sqrt((X * X).sum(axis=axis))

    # element-wise -------------------------------------------------------------------
    abs = staticmethod(np.abs)
    sign = staticmethod(np.sign)
    sqrt = staticmethod(np.sqrt)
    exp = staticmethod(np.exp)
    isfinite = staticmethod(np.isfinite)

    @staticmethod
    def minimum(a, b):
        return np.minimum(a, b)

    @staticmethod
    def maximum(a, b):
        return np.maximum(a, b)

    @staticmethod
    def clip(x, lo, hi):
        return np.clip(x, lo, hi)

    @staticmethod
    def where(cond, a, b):
        return np.where(cond, a, b)

    # constructors -------------------------------------------------------------------
    @staticmethod
    def zeros_like(x):
        return np.zeros_like(x)

    @staticmethod
    def ones_like(x):
        return np.ones_like(x)

    @staticmethod
    def full_like(x, value):
        return np.full_like(x, value)

    @staticmethod
    def zeros(shape, like):
        return np.zeros(shape, dtype=like.dtype)

    @staticmethod
    def asarray(values, like):
        return np.asarray(values, dtype=like.dtype)

    @staticmethod
    def index(values, like):
        """Integer index array usable to index ``like``."""
        return np.asarray(values, dtype=np.int64)

    @staticmethod
    def stack(vectors, axis=0):
        return np.stack(vectors, axis=axis)

    @staticmethod
    def concat(vectors, axis=0):
        return np.concatenate(vectors, axis=axis)

    @staticmethod
    def copy(x):
        return x.copy()

    # random -------------------------------------------------------------------------
    @staticmethod
    def randn_like(x, generator=None):
        rng = generator if generator is not None else np.random
        return rng.standard_normal(size=x.shape).astype(x.dtype, copy=False)

    @staticmethod
    def permutation(n, like, generator=None):
        rng = generator if generator is not None else np.random
        return rng.permutation(n)

    # misc ---------------------------------------------------------------------------
    @staticmethod
    def take_along_axis(X, idx, axis=0):
        return np.take_along_axis(X, idx, axis=axis)

    @staticmethod
    def dot(a, b):
        return float(np.dot(a, b))

    @staticmethod
    def item(x) -> float:
        return float(x)


# ----------------------------------------------------------------------------------
# PyTorch implementation (same API)
# ----------------------------------------------------------------------------------
class _TorchOps:
    name = "torch"

    @staticmethod
    def mean(X, axis=0):
        return X.mean(dim=axis)

    @staticmethod
    def sum(X, axis=0):
        return X.sum(dim=axis)

    @staticmethod
    def std(X, axis=0):
        return X.std(dim=axis, correction=1)

    @staticmethod
    def sort(X, axis=0):
        return torch.sort(X, dim=axis).values

    @staticmethod
    def argsort(X, axis=-1):
        return torch.argsort(X, dim=axis, stable=True)

    @staticmethod
    def median(X, axis=0):
        return _median_from_sorted(torch.sort(X, dim=axis).values, axis)

    @staticmethod
    def norm(X, axis=-1):
        return torch.sqrt((X * X).sum(dim=axis))

    abs = staticmethod(torch.abs)
    sign = staticmethod(torch.sign)
    sqrt = staticmethod(torch.sqrt)
    exp = staticmethod(torch.exp)
    isfinite = staticmethod(torch.isfinite)

    @staticmethod
    def minimum(a, b):
        if not is_torch(b):
            b = torch.as_tensor(b, dtype=a.dtype, device=a.device)
        if not is_torch(a):
            a = torch.as_tensor(a, dtype=b.dtype, device=b.device)
        return torch.minimum(a, b)

    @staticmethod
    def maximum(a, b):
        if not is_torch(b):
            b = torch.as_tensor(b, dtype=a.dtype, device=a.device)
        if not is_torch(a):
            a = torch.as_tensor(a, dtype=b.dtype, device=b.device)
        return torch.maximum(a, b)

    @staticmethod
    def clip(x, lo, hi):
        return torch.clamp(x, lo, hi)

    @staticmethod
    def where(cond, a, b):
        return torch.where(cond, a, b)

    @staticmethod
    def zeros_like(x):
        return torch.zeros_like(x)

    @staticmethod
    def ones_like(x):
        return torch.ones_like(x)

    @staticmethod
    def full_like(x, value):
        return torch.full_like(x, value)

    @staticmethod
    def zeros(shape, like):
        return torch.zeros(shape, dtype=like.dtype, device=like.device)

    @staticmethod
    def asarray(values, like):
        return torch.as_tensor(values, dtype=like.dtype, device=like.device)

    @staticmethod
    def index(values, like):
        return torch.as_tensor(values, dtype=torch.long, device=like.device)

    @staticmethod
    def stack(vectors, axis=0):
        return torch.stack(list(vectors), dim=axis)

    @staticmethod
    def concat(vectors, axis=0):
        return torch.cat(list(vectors), dim=axis)

    @staticmethod
    def copy(x):
        return x.clone()

    @staticmethod
    def randn_like(x, generator=None):
        return torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)

    @staticmethod
    def permutation(n, like, generator=None):
        return torch.randperm(n, generator=generator).to(like.device)

    @staticmethod
    def take_along_axis(X, idx, axis=0):
        return torch.gather(X, axis, idx)

    @staticmethod
    def dot(a, b):
        return float(torch.dot(a, b))

    @staticmethod
    def item(x) -> float:
        return float(x)


NUMPY = _NumpyOps()
TORCH = _TorchOps()


def _median_from_sorted(S, axis):
    """Median of an already sorted array along ``axis`` (average of the two middles)."""
    n = S.shape[axis]
    if axis != 0:
        raise ValueError("flcourse only uses median along axis 0")
    mid = n // 2
    if n % 2 == 1:
        return S[mid]
    return (S[mid - 1] + S[mid]) / 2


# ----------------------------------------------------------------------------------
# Public helpers
# ----------------------------------------------------------------------------------
def ops_for(x: Array):
    """Return the ops namespace (NUMPY or TORCH) matching ``x``."""
    if isinstance(x, (list, tuple)):
        x = x[0]
    return TORCH if is_torch(x) else NUMPY


def as_matrix(vectors: Array | Sequence[Array]) -> Tuple[Array, Any]:
    """Turn ``vectors`` into a floating point ``(n, d)`` matrix and return ``(X, xp)``.

    Accepts a 2-D numpy array / torch tensor or a list of 1-D vectors (any shape is
    flattened). Never modifies the input in place.
    """
    if isinstance(vectors, (list, tuple)):
        if len(vectors) == 0:
            raise ValueError("Need at least one vector.")
        if is_torch(vectors[0]):
            X = torch.stack([v.reshape(-1) for v in vectors])
        else:
            X = np.stack([np.asarray(v).reshape(-1) for v in vectors])
    else:
        X = vectors

    if is_torch(X):
        if X.ndim != 2:
            raise ValueError(f"Expected a (n, d) matrix, got shape {tuple(X.shape)}.")
        if not torch.is_floating_point(X):
            X = X.float()
        return X, TORCH

    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Expected a (n, d) matrix, got shape {X.shape}.")
    if not np.issubdtype(X.dtype, np.floating):
        X = X.astype(np.float64)
    return X, NUMPY


def as_vector(vector: Array) -> Tuple[Array, Any]:
    """Return ``(v, xp)`` with ``v`` a flattened floating point 1-D vector."""
    if is_torch(vector):
        v = vector.reshape(-1)
        if not torch.is_floating_point(v):
            v = v.float()
        return v, TORCH
    v = np.asarray(vector).reshape(-1)
    if not np.issubdtype(v.dtype, np.floating):
        v = v.astype(np.float64)
    return v, NUMPY


def stack(vectors: Sequence[Array]) -> Array:
    """Stack a list of 1-D vectors into an ``(n, d)`` matrix (numpy or torch)."""
    return as_matrix(vectors)[0]


def to_numpy(x: Array) -> np.ndarray:
    """Convert a torch tensor (any device) or array-like to a numpy array."""
    if is_torch(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def to_torch(x: Array, like: torch.Tensor | None = None, device=None, dtype=None) -> torch.Tensor:
    """Convert to a torch tensor; ``like`` copies dtype and device from a reference tensor."""
    if like is not None:
        device = like.device if device is None else device
        dtype = like.dtype if dtype is None else dtype
    if is_torch(x):
        return x.to(device=device or x.device, dtype=dtype or x.dtype)
    t = torch.as_tensor(np.asarray(x))
    return t.to(device=device or "cpu", dtype=dtype or torch.float32)


def to_backend(x: Array, backend: str, like: torch.Tensor | None = None) -> Array:
    """Convert ``x`` to ``backend`` ("numpy" or "torch")."""
    if backend == "numpy":
        return to_numpy(x)
    if backend == "torch":
        return to_torch(x, like=like)
    raise ValueError(f"Unknown backend {backend!r} (use 'numpy' or 'torch').")


def same_type_as(result: Array, reference: Array) -> Array:
    """Convert ``result`` to the library/device/dtype of ``reference``."""
    if is_torch(reference):
        return to_torch(result, like=reference)
    out = to_numpy(result)
    ref = np.asarray(reference)
    return out.astype(ref.dtype, copy=False) if np.issubdtype(ref.dtype, np.floating) else out


def pairwise_sq_distances(X: Array, xp=None) -> Array:
    """Squared Euclidean distances between all rows of ``X``: ``D[i, j] = ||x_i - x_j||^2``.

    Implemented with an explicit loop over rows: memory stays O(n*d) (no (n, n, d)
    tensor) and there is no cancellation error as with the ``|x|^2 + |y|^2 - 2xy`` trick.
    """
    xp = xp or ops_for(X)
    n = X.shape[0]
    D = xp.zeros((n, n), like=X)
    for i in range(n):
        diff = X - X[i]
        D[i] = xp.sum(diff * diff, axis=1)
    return D
