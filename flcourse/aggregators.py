"""Aggregators

The main interface is the python macro __call__ with one main parameter: "vectors"

    An example of one of the aggregators from below:

    def __call__(self, vectors, weights=None, reference=None):

        return xp.mean(...)

    You can call this with
        aggregate = Krum(f=2)(vectors)

    ``vectors`` is an ``(n, d)`` numpy array / torch tensor or a list of 1-D vectors; the
    result is a ``(d,)`` vector of the same library (see :mod:`flcourse.backend`).

    Some aggregators also need per-client weights, e.g. number of samples (``WeightedAverage``) and some (e.g. FLtrust) need a reference dataset

    Aggregators ignore side information they do not need, so the round loop can always
    call ``aggregator(X, weights=..., reference=...)``.

    
HOW TO MODIFY: 

    Adding a new rule = write a class with ``__call__`` and add it to ``AGGREGATORS`` at the
    bottom of this file. Then ``aggregator: {name: my_rule, ...}`` works in every config.

    Notation below: ``n`` vectors, at most ``f`` of them Byzantine.

    Note on secure aggregation: every rule except (weighted) averaging needs to see the
    *individual* vectors, which secure aggregation hides on purpose. Robustness and input
    privacy are therefore not simply composable. If you try to do this in the config yaml,
    an exception is thrown.

"""
from __future__ import annotations

import inspect
from typing import Optional

from .backend import as_matrix, as_vector, pairwise_sq_distances, same_type_as


class Aggregator:
    """Base class. ``needs_reference = True`` tells the round loop to compute a server update."""

    needs_reference = False

    def __call__(self, vectors, weights=None, reference=None):  # pragma: no cover - interface
        raise NotImplementedError

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in vars(self).items() if not k.startswith("_"))
        return f"{type(self).__name__}({params})"


def _check_f(n: int, f: int, need: int, rule: str, formula: str) -> None:
    if f < 0:
        raise ValueError(f"{rule}: f must be >= 0")
    if n < need:
        raise ValueError(f"{rule} with f={f} needs n >= {formula} = {need} vectors, got n={n}.")


# ----------------------------------------------------------------------------------
# averaging
# ----------------------------------------------------------------------------------
class Average(Aggregator):
    """Plain mean -- FedAvg with equal client weights. Not robust: one vector can move it arbitrarily."""

    def __call__(self, vectors, weights=None, reference=None):
        X, xp = as_matrix(vectors)
        return xp.mean(X, axis=0)


class WeightedAverage(Aggregator):
    """FedAvg (McMahan et al. 2017): mean weighted by ``weights`` (e.g. local dataset sizes)."""

    def __call__(self, vectors, weights=None, reference=None):
        X, xp = as_matrix(vectors)
        if weights is None:
            return xp.mean(X, axis=0)
        w = xp.asarray(weights, like=X).reshape(-1, 1)
        return xp.sum(w * X, axis=0) / xp.sum(w, axis=0)


# ----------------------------------------------------------------------------------
# coordinate-wise rules
# ----------------------------------------------------------------------------------
class Median(Aggregator):
    """Coordinate-wise median (Yin et al., ICML 2018)."""

    def __call__(self, vectors, weights=None, reference=None):
        X, xp = as_matrix(vectors)
        return xp.median(X, axis=0)


class TrimmedMean(Aggregator):
    """Coordinate-wise trimmed mean (Yin et al., ICML 2018): per coordinate drop the ``f``
    largest and ``f`` smallest values and average the rest. Needs ``n > 2f``."""

    def __init__(self, f: int = 1):
        self.f = int(f)

    def __call__(self, vectors, weights=None, reference=None):
        X, xp = as_matrix(vectors)
        n = X.shape[0]
        _check_f(n, self.f, 2 * self.f + 1, "TrimmedMean", "2f+1")
        S = xp.sort(X, axis=0)
        return xp.mean(S[self.f:n - self.f], axis=0)


# ----------------------------------------------------------------------------------
# geometric rules
# ----------------------------------------------------------------------------------
class GeometricMedian(Aggregator):
    """Geometric median via the smoothed Weiszfeld algorithm (RFA, Pillutla et al. 2022):

        z <- sum_i w_i x_i / sum_i w_i   with   w_i = 1 / max(nu, ||x_i - z||)
    """

    def __init__(self, nu: float = 1e-6, max_iter: int = 100, tol: float = 1e-6):
        self.nu, self.max_iter, self.tol = float(nu), int(max_iter), float(tol)

    def __call__(self, vectors, weights=None, reference=None):
        X, xp = as_matrix(vectors)
        z = xp.mean(X, axis=0)
        for _ in range(self.max_iter):
            dist = xp.norm(X - z, axis=1)
            w = 1.0 / xp.maximum(dist, self.nu)
            z_new = xp.sum(w.reshape(-1, 1) * X, axis=0) / xp.sum(w, axis=0)
            moved = xp.item(xp.norm(z_new - z))
            z = z_new
            if moved <= self.tol * max(xp.item(xp.norm(z)), 1e-12):
                break
        return z


def krum_scores(X, f: int, xp):
    """Krum score of every vector: sum of squared distances to its ``n - f - 2`` nearest neighbours."""
    n = X.shape[0]
    k = max(n - f - 2, 1)
    D = pairwise_sq_distances(X, xp)
    D_sorted = xp.sort(D, axis=1)               # column 0 is the distance to itself (0)
    return xp.sum(D_sorted[:, 1:k + 1], axis=1)


class Krum(Aggregator):
    """Krum (Blanchard et al., NeurIPS 2017): return the single vector with the smallest
    Krum score. Needs ``n >= 2f + 3``."""

    def __init__(self, f: int = 1):
        self.f = int(f)

    def __call__(self, vectors, weights=None, reference=None):
        X, xp = as_matrix(vectors)
        _check_f(X.shape[0], self.f, 2 * self.f + 3, "Krum", "2f+3")
        scores = krum_scores(X, self.f, xp)
        best = int(xp.argsort(scores, axis=0)[0])
        return X[best]


class MultiKrum(Aggregator):
    """Multi-Krum: average of the ``m`` vectors with the smallest Krum scores (default ``m = n - f``)."""

    def __init__(self, f: int = 1, m: Optional[int] = None):
        self.f, self.m = int(f), m

    def __call__(self, vectors, weights=None, reference=None):
        X, xp = as_matrix(vectors)
        n = X.shape[0]
        _check_f(n, self.f, 2 * self.f + 3, "MultiKrum", "2f+3")
        m = self.m if self.m is not None else n - self.f
        order = xp.argsort(krum_scores(X, self.f, xp), axis=0)[:m]
        return xp.mean(X[order], axis=0)


class Bulyan(Aggregator):
    """Bulyan (El Mhamdi et al., ICML 2018). Needs ``n >= 4f + 3``.

    1. select ``theta = n - 2f`` vectors by repeatedly applying Krum to the remaining ones;
    2. per coordinate, average the ``beta = theta - 2f`` values closest to the median.
    """

    def __init__(self, f: int = 1):
        self.f = int(f)

    def __call__(self, vectors, weights=None, reference=None):
        X, xp = as_matrix(vectors)
        n, f = X.shape[0], self.f
        _check_f(n, f, 4 * f + 3, "Bulyan", "4f+3")
        remaining = list(range(n))
        selected = []
        for _ in range(n - 2 * f):
            sub = X[xp.index(remaining, X)]
            best = int(xp.argsort(krum_scores(sub, f, xp), axis=0)[0])
            selected.append(remaining.pop(best))
        S = X[xp.index(selected, X)]
        beta = len(selected) - 2 * f
        med = xp.median(S, axis=0)
        closest = xp.argsort(xp.abs(S - med), axis=0)[:beta]
        return xp.mean(xp.take_along_axis(S, closest, axis=0), axis=0)


class CenteredClipping(Aggregator):
    """Centered clipping (Karimireddy et al., ICML 2021), as in ByzFL:

        v <- v + mean_i (x_i - v) * min(1, tau / ||x_i - v||)      (L iterations)

    The centre ``v`` starts at the previous output (zero in the first round), i.e. the
    aggregator is *stateful*. Meaningful for ``upload: delta`` or ``gradient``.
    """

    def __init__(self, tau: float = 10.0, L: int = 1):
        self.tau, self.L = float(tau), int(L)
        self._center = None

    def reset(self) -> None:
        self._center = None

    def __call__(self, vectors, weights=None, reference=None):
        X, xp = as_matrix(vectors)
        v = xp.zeros_like(X[0]) if self._center is None else same_type_as(self._center, X[0])
        for _ in range(self.L):
            diff = X - v
            norms = xp.norm(diff, axis=1)
            scale = xp.minimum(xp.ones_like(norms), self.tau / xp.maximum(norms, 1e-12))
            v = v + xp.mean(diff * scale.reshape(-1, 1), axis=0)
        self._center = xp.copy(v)
        return v


class FLTrust(Aggregator):
    """FLTrust (Cao et al., NDSS 2021). The server trains on a small clean *root* dataset
    and obtains its own update ``r`` (``reference``). Each client gets the trust score
    ``ReLU(cos(x_i, r))``, its vector is rescaled to ``||r||``, and the result is the
    trust-weighted average. Use with ``upload: delta`` or ``gradient``.
    """

    needs_reference = True

    def __call__(self, vectors, weights=None, reference=None):
        if reference is None:
            raise ValueError("FLTrust needs the server update: aggregator(X, reference=server_update)")
        X, xp = as_matrix(vectors)
        r, _ = as_vector(reference)
        r = same_type_as(r, X[0])
        r_norm = xp.norm(r)
        x_norms = xp.maximum(xp.norm(X, axis=1), 1e-12)
        cos = xp.sum(X * r, axis=1) / (x_norms * xp.maximum(r_norm, 1e-12))
        trust = xp.maximum(cos, xp.zeros_like(cos))                          # ReLU
        if xp.item(xp.sum(trust, axis=0)) <= 0:
            return xp.zeros_like(r)                                             # nobody is trusted
        rescaled = X * (r_norm / x_norms).reshape(-1, 1)
        return xp.sum(trust.reshape(-1, 1) * rescaled, axis=0) / xp.sum(trust, axis=0)


# ----------------------------------------------------------------------------------
# registry
# ----------------------------------------------------------------------------------
AGGREGATORS = {
    "mean": Average,
    "average": Average,
    "weighted_mean": WeightedAverage,
    "fedavg": WeightedAverage,
    "median": Median,
    "trimmed_mean": TrimmedMean,
    "trmean": TrimmedMean,
    "geometric_median": GeometricMedian,
    "geomed": GeometricMedian,
    "rfa": GeometricMedian,
    "krum": Krum,
    "multi_krum": MultiKrum,
    "multikrum": MultiKrum,
    "bulyan": Bulyan,
    "centered_clipping": CenteredClipping,
    "cclip": CenteredClipping,
    "fltrust": FLTrust,
}


def _instantiate(registry: dict, spec, f_default: Optional[int], kind: str):
    if spec is None:
        spec = {"name": "mean"} if kind == "aggregator" else {}
    if isinstance(spec, str):
        spec = {"name": spec}
    spec = dict(spec)
    name = str(spec.pop("name")).lower()
    if name not in registry:
        raise ValueError(f"Unknown {kind} {name!r}. Available: {sorted(registry)}")
    cls = registry[name]
    params = inspect.signature(cls.__init__).parameters
    if "f" in params and "f" not in spec and f_default is not None:
        spec["f"] = int(f_default)
    unknown = set(spec) - set(params)
    if unknown:
        raise ValueError(f"{cls.__name__} got unknown parameters {sorted(unknown)}; accepts {list(params)[1:]}")
    return cls(**spec)


def make_aggregator(spec, f_default: Optional[int] = None) -> Aggregator:
    """Build an aggregator from a config entry like ``{"name": "krum", "f": 2}``.

    If the rule takes ``f`` and the config does not set it, ``f_default`` is used (the
    experiments pass the number of Byzantine clients).
    """
    return _instantiate(AGGREGATORS, spec, f_default, "aggregator")
