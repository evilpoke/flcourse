"""Pre-aggregators: transform the matrix of uploads *before* the aggregator (ByzFL style).

    X = NNM(f=2)(X)              # (n, d) -> (n, d)
    aggregate = TrimmedMean(f=2)(X)

All of them return a new matrix of the same library (inputs are never modified in place).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from .aggregators import _instantiate
from .backend import as_matrix, pairwise_sq_distances


class PreAggregator:
    def __call__(self, vectors):  # pragma: no cover - interface
        raise NotImplementedError

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in vars(self).items() if not k.startswith("_"))
        return f"{type(self).__name__}({params})"


class NNM(PreAggregator):
    """Nearest Neighbour Mixing (Allouah et al., AISTATS 2023): replace every vector by the
    average of its ``n - f`` nearest neighbours (itself included). Reduces heterogeneity
    before a robust aggregator."""

    def __init__(self, f: int = 1):
        self.f = int(f)

    def __call__(self, vectors):
        X, xp = as_matrix(vectors)
        n = X.shape[0]
        if not 0 <= self.f < n:
            raise ValueError(f"NNM: need 0 <= f < n, got f={self.f}, n={n}")
        k = n - self.f
        nearest = xp.argsort(pairwise_sq_distances(X, xp), axis=1)[:, :k]
        return xp.stack([xp.mean(X[nearest[i]], axis=0) for i in range(n)])


class Bucketing(PreAggregator):
    """Bucketing (Karimireddy et al., ICLR 2022): shuffle, split into buckets of ``s``
    vectors and replace every bucket by its mean (output has ``ceil(n/s)`` rows)."""

    def __init__(self, s: int = 2, seed: Optional[int] = 0):
        self.s = int(s)
        self.seed = seed
        self._np_rng = np.random.default_rng(seed)
        self._torch_rng = torch.Generator().manual_seed(seed if seed is not None else 0)

    def __call__(self, vectors):
        X, xp = as_matrix(vectors)
        n = X.shape[0]
        rng = self._torch_rng if xp.name == "torch" else self._np_rng
        perm = xp.permutation(n, like=X, generator=rng)
        X = X[perm]
        return xp.stack([xp.mean(X[i:i + self.s], axis=0) for i in range(0, n, self.s)])


class Clipping(PreAggregator):
    """Static l2 clipping: every vector with norm > c is scaled down to norm c."""

    def __init__(self, c: float = 2.0):
        self.c = float(c)

    def __call__(self, vectors):
        X, xp = as_matrix(vectors)
        norms = xp.norm(X, axis=1)
        scale = xp.minimum(xp.ones_like(norms), self.c / xp.maximum(norms, 1e-12))
        return X * scale.reshape(-1, 1)


class ARC(PreAggregator):
    """Adaptive Robust Clipping (Allouah et al., 2024): clip the ``k = floor(2 f (n-f) / n)``
    largest vectors to the norm of the largest *unclipped* one."""

    def __init__(self, f: int = 1):
        self.f = int(f)

    def __call__(self, vectors):
        X, xp = as_matrix(vectors)
        n = X.shape[0]
        k = int(math.floor(2 * self.f * (n - self.f) / n))
        if k <= 0:
            return xp.copy(X)
        norms = xp.norm(X, axis=1)
        threshold = xp.sort(norms, axis=0)[n - k - 1]
        scale = xp.minimum(xp.ones_like(norms), threshold / xp.maximum(norms, 1e-12))
        return X * scale.reshape(-1, 1)


PRE_AGGREGATORS = {
    "nnm": NNM,
    "bucketing": Bucketing,
    "clipping": Clipping,
    "arc": ARC,
}


def make_pre_aggregators(specs, f_default: Optional[int] = None):
    """Build the list of pre-aggregators from a config list like ``[{"name": "nnm"}]``."""
    return [_instantiate(PRE_AGGREGATORS, spec, f_default, "pre-aggregator") for spec in (specs or [])]
