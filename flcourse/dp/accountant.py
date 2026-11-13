"""Rényi-DP accountant for the (Poisson-)subsampled Gaussian mechanism.

Used for client-level DP: each round, every client is sampled with probability
``q = clients_per_round / num_clients`` and the (clipped) sum is released with noise
multiplier ``sigma``. After ``T`` rounds::

    acc = RDPAccountant()
    acc.step(noise_multiplier=sigma, sample_rate=q, num_steps=T)
    eps = acc.get_epsilon(delta=1e-5)

The RDP of one step is computed exactly for integer orders alpha (Mironov et al. 2019,
"Rényi DP of the sampled Gaussian mechanism", Sec. 3.3)::

    A_alpha = sum_k  C(alpha, k) (1-q)^(alpha-k) q^k  exp((k^2 - k) / (2 sigma^2))
    RDP(alpha) = log(A_alpha) / (alpha - 1)

and converted to (epsilon, delta)-DP with the bound of Balle et al. 2020 (also used by
Opacus): ``eps = RDP + log((alpha-1)/alpha) - (log delta + log alpha) / (alpha - 1)``.
Only integer orders are used, so the result is very slightly looser than Opacus (which
also evaluates fractional orders) -- see ``tests/test_dp.py`` for the comparison.

Note: the bound assumes Poisson sampling; the experiments sample a fixed number of
clients per round, which is the usual approximation in FL papers.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np

DEFAULT_ORDERS: List[int] = list(range(2, 129)) + [160, 192, 256, 320, 384, 512, 768, 1024]


def _log_add(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    hi, lo = max(a, b), min(a, b)
    return hi + math.log1p(math.exp(lo - hi))


def _log_binom(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def rdp_subsampled_gaussian(q: float, sigma: float, orders: Sequence[int] = DEFAULT_ORDERS) -> np.ndarray:
    """RDP of ONE step of the subsampled Gaussian mechanism at integer ``orders``."""
    orders = [int(a) for a in orders]
    if sigma <= 0:
        return np.full(len(orders), np.inf)
    if q <= 0:
        return np.zeros(len(orders))
    if q >= 1:
        return np.array([a / (2 * sigma**2) for a in orders])
    out = []
    log_q, log_1mq = math.log(q), math.log1p(-q)
    for a in orders:
        log_a = -math.inf
        for k in range(a + 1):
            term = _log_binom(a, k) + k * log_q + (a - k) * log_1mq + (k * k - k) / (2 * sigma**2)
            log_a = _log_add(log_a, term)
        out.append(log_a / (a - 1))
    return np.array(out)


def rdp_to_epsilon(rdp: Sequence[float], orders: Sequence[int], delta: float) -> Tuple[float, int]:
    """Best (epsilon, order) for a target ``delta``."""
    rdp, orders_arr = np.asarray(rdp, dtype=float), np.asarray(orders, dtype=float)
    eps = rdp - (math.log(delta) + np.log(orders_arr)) / (orders_arr - 1) + np.log((orders_arr - 1) / orders_arr)
    eps = np.where(np.isnan(eps), np.inf, eps)
    i = int(np.argmin(eps))
    return max(float(eps[i]), 0.0), int(orders_arr[i])


class RDPAccountant:
    """Accumulates RDP over (possibly heterogeneous) steps."""

    def __init__(self, orders: Iterable[int] = DEFAULT_ORDERS):
        self.orders = list(orders)
        self.rdp = np.zeros(len(self.orders))
        self.history: List[Tuple[float, float, int]] = []

    def step(self, noise_multiplier: float, sample_rate: float, num_steps: int = 1) -> None:
        self.rdp = self.rdp + num_steps * rdp_subsampled_gaussian(sample_rate, noise_multiplier, self.orders)
        self.history.append((noise_multiplier, sample_rate, num_steps))

    def get_privacy_spent(self, delta: float) -> Tuple[float, int]:
        return rdp_to_epsilon(self.rdp, self.orders, delta)

    def get_epsilon(self, delta: float) -> float:
        return self.get_privacy_spent(delta)[0]


def epsilon_after(noise_multiplier: float, sample_rate: float, steps: int, delta: float) -> float:
    acc = RDPAccountant()
    acc.step(noise_multiplier, sample_rate, steps)
    return acc.get_epsilon(delta)


def noise_multiplier_for(target_epsilon: float, delta: float, sample_rate: float, steps: int,
                         tol: float = 0.01) -> float:
    """Smallest noise multiplier reaching ``target_epsilon`` after ``steps`` (bisection)."""
    lo, hi = 0.0, 1.0
    while epsilon_after(hi, sample_rate, steps, delta) > target_epsilon:
        lo, hi = hi, 2 * hi
        if hi > 1e4:
            raise ValueError("target epsilon not reachable")
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if epsilon_after(mid, sample_rate, steps, delta) > target_epsilon:
            lo = mid
        else:
            hi = mid
    return hi
