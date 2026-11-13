"""Model-poisoning attacks on uploaded vectors (ByzFL interface).

    byzantine_vector = ALittleIsEnough(tau=1.5)(honest_vectors)

``honest_vectors`` is whatever the colluding attackers know about honest uploads
(an ``(n, d)`` matrix or a list of vectors): in the *non-omniscient* setting these are
the honest vectors the attackers computed on their own data and shared on the
collusion channel; in the *omniscient* setting all honest uploads of the round.
All Byzantine clients then submit the returned vector.

Data-poisoning attacks (label flipping, backdoors) live in ``data_poisoning.py``:
they change the local data of a malicious client, not its vector.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..backend import as_matrix, pairwise_sq_distances


class Attack:
    def __call__(self, honest_vectors):  # pragma: no cover - interface
        raise NotImplementedError

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in vars(self).items() if not k.startswith("_"))
        return f"{type(self).__name__}({params})"


class SignFlipping(Attack):
    """``-mean(honest)`` (Allen-Zhu et al. 2020)."""

    def __call__(self, honest_vectors):
        X, xp = as_matrix(honest_vectors)
        return -xp.mean(X, axis=0)


class InnerProductManipulation(Attack):
    """IPM (Xie et al., UAI 2020): ``-tau * mean(honest)``; makes the inner product of the
    aggregate with the true gradient negative for large enough ``tau``."""

    def __init__(self, tau: float = 2.0):
        self.tau = float(tau)

    def __call__(self, honest_vectors):
        X, xp = as_matrix(honest_vectors)
        return -self.tau * xp.mean(X, axis=0)


class ALittleIsEnough(Attack):
    """ALIE (Baruch et al., NeurIPS 2019): ``mean + tau * std`` (coordinate-wise, std with
    ddof=1). Small, coordinated shifts that stay inside the spread of honest vectors and
    therefore evade median/Krum-style defenses."""

    def __init__(self, tau: float = 1.5):
        self.tau = float(tau)

    def __call__(self, honest_vectors):
        X, xp = as_matrix(honest_vectors)
        if X.shape[0] < 2:        # a single attacker cannot estimate the spread -> behaves honestly
            return xp.mean(X, axis=0)
        return xp.mean(X, axis=0) + self.tau * xp.std(X, axis=0)


class Gaussian(Attack):
    """Random vector with i.i.d. ``N(mu, sigma^2)`` coordinates (FedML's "random" mode)."""

    def __init__(self, mu: float = 0.0, sigma: float = 1.0, seed: Optional[int] = None):
        self.mu, self.sigma, self.seed = float(mu), float(sigma), seed
        self._np = np.random.default_rng(seed)
        self._torch = torch.Generator().manual_seed(seed if seed is not None else 0)

    def __call__(self, honest_vectors):
        X, xp = as_matrix(honest_vectors)
        if xp.name == "torch":
            noise = torch.randn(X.shape[1], generator=self._torch).to(dtype=X.dtype, device=X.device)
        else:
            noise = self._np.standard_normal(X.shape[1]).astype(X.dtype)
        return self.mu + self.sigma * noise


class Mimic(Attack):
    """Copy one honest vector (Karimireddy et al., ICLR 2022): over-represents one client,
    harmful under heterogeneity."""

    def __init__(self, target: int = 0):
        self.target = int(target)

    def __call__(self, honest_vectors):
        X, xp = as_matrix(honest_vectors)
        return xp.copy(X[self.target % X.shape[0]])


class Inf(Attack):
    """Vector of ``+inf`` (breaks averaging immediately)."""

    def __call__(self, honest_vectors):
        X, xp = as_matrix(honest_vectors)
        return xp.full_like(X[0], float("inf"))


class Zero(Attack):
    """All-zero vector (FedML's "zero" mode; a lazy / free-riding client in delta mode)."""

    def __call__(self, honest_vectors):
        X, xp = as_matrix(honest_vectors)
        return xp.zeros_like(X[0])


class Scaling(Attack):
    """Boost the attackers' own (honest-looking or backdoored) mean by ``factor`` --
    the model-replacement attack of Bagdasaryan et al. (AISTATS 2020) in ``delta`` mode."""

    def __init__(self, factor: float = 10.0):
        self.factor = float(factor)

    def __call__(self, honest_vectors):
        X, xp = as_matrix(honest_vectors)
        return self.factor * xp.mean(X, axis=0)


class MinMax(Attack):
    """Min-Max attack (Shejwalkar & Houmansadr, NDSS 2021), aggregation-rule agnostic.

    ``mal = mean - gamma * p`` where ``p`` is the coordinate-wise std (``"std"``), the unit
    vector of the mean (``"unit"``) or its sign (``"sign"``). ``gamma`` is found by binary
    search as the largest value such that the maximum distance of ``mal`` to any honest
    vector is not larger than the maximum distance between two honest vectors.
    """

    objective = "max"

    def __init__(self, perturbation: str = "std", gamma_init: float = 10.0, tol: float = 1e-5):
        self.perturbation, self.gamma_init, self.tol = perturbation, float(gamma_init), float(tol)

    def _direction(self, X, xp):
        mean = xp.mean(X, axis=0)
        if self.perturbation == "std":
            return xp.std(X, axis=0)
        if self.perturbation == "unit":
            return mean / xp.maximum(xp.norm(mean), 1e-12)
        if self.perturbation == "sign":
            return xp.sign(mean)
        raise ValueError(f"unknown perturbation {self.perturbation!r}")

    def _score(self, mal, X, xp) -> float:
        d = xp.sum((X - mal) ** 2, axis=1)
        return xp.item(xp.sum(d, axis=0) if self.objective == "sum" else xp.sort(d, axis=0)[-1])

    def _threshold(self, X, xp) -> float:
        D = pairwise_sq_distances(X, xp)
        per_row = xp.sum(D, axis=1) if self.objective == "sum" else xp.sort(D, axis=1)[:, -1]
        return xp.item(xp.sort(per_row, axis=0)[-1])

    def __call__(self, honest_vectors):
        X, xp = as_matrix(honest_vectors)
        if X.shape[0] < 2:        # no reference distances available -> behaves honestly
            return xp.mean(X, axis=0)
        mean, p = xp.mean(X, axis=0), self._direction(X, xp)
        threshold = self._threshold(X, xp)
        # binary search exactly as in the authors' code: gamma moves by step/2, step halves.
        # Starting with step = gamma_init lets gamma reach any value in (0, 2 * gamma_init).
        gamma, step, gamma_ok = self.gamma_init, self.gamma_init, 0.0
        while abs(gamma_ok - gamma) > self.tol:
            if self._score(mean - gamma * p, X, xp) <= threshold:
                gamma_ok = gamma
                gamma = gamma + step / 2
            else:
                gamma = gamma - step / 2
            step = step / 2
        return mean - gamma_ok * p


class MinSum(MinMax):
    """Min-Sum variant: the *sum* of squared distances to all honest vectors may not
    exceed the largest such sum of an honest vector."""

    objective = "sum"


ATTACKS = {
    "sign_flipping": SignFlipping,
    "signflip": SignFlipping,
    "ipm": InnerProductManipulation,
    "inner_product_manipulation": InnerProductManipulation,
    "alie": ALittleIsEnough,
    "a_little_is_enough": ALittleIsEnough,
    "gaussian": Gaussian,
    "random": Gaussian,
    "mimic": Mimic,
    "inf": Inf,
    "zero": Zero,
    "scaling": Scaling,
    "model_replacement": Scaling,
    "minmax": MinMax,
    "min_max": MinMax,
    "minsum": MinSum,
    "min_sum": MinSum,
}


def make_attack(spec) -> Optional[Attack]:
    """Build a vector attack from a config entry, e.g. ``{"name": "alie", "tau": 1.5}``.
    ``None`` / ``"none"`` returns ``None`` (attackers submit what they computed)."""
    if spec is None:
        return None
    if isinstance(spec, str):
        spec = {"name": spec}
    spec = dict(spec)
    name = str(spec.pop("name", "none")).lower()
    if name in ("none", "", "null"):
        return None
    if name not in ATTACKS:
        raise ValueError(f"Unknown attack {name!r}. Available: {sorted(ATTACKS)}")
    return ATTACKS[name](**spec)
