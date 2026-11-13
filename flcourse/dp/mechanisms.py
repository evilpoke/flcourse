"""Differential-privacy building blocks (work on numpy arrays and torch tensors).

Client-level DP for federated learning ("DP-FedAvg", McMahan et al., ICLR 2018) is built
from two one-liners in the round loop::

    upload = clip_l2(upload, max_norm=C)                   # bound each client's influence
    upload = add_gaussian_noise(upload, std=sigma * C)     # local DP: every client adds noise

or, with a trusted server (central DP), the server adds ``N(0, (sigma C)^2)`` to the
*sum* of the clipped uploads. ``sigma`` is the *noise multiplier*; the privacy loss is
computed by :mod:`flcourse.dp.accountant`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..backend import as_vector, is_torch


def l2_norm(x) -> float:
    v, xp = as_vector(x)
    return xp.item(xp.norm(v))


def clip_l2(x, max_norm: float):
    """Scale ``x`` down to l2 norm ``max_norm`` if it is longer (same shape and type)."""
    v, xp = as_vector(x)
    norm = xp.item(xp.norm(v))
    if norm <= max_norm or norm == 0.0:
        return x
    return x * (max_norm / norm)


def add_gaussian_noise(x, std: float, generator=None):
    """Return ``x + N(0, std^2 I)``. ``generator``: a ``torch.Generator`` (torch input) or a
    ``numpy.random.Generator`` (numpy input); ``None`` uses the global RNG."""
    if std == 0:
        return x
    if is_torch(x):
        noise = torch.randn(x.shape, generator=generator, dtype=torch.float32).to(device=x.device, dtype=x.dtype) \
            if generator is not None else torch.randn_like(x)
        return x + std * noise
    rng = generator if generator is not None else np.random.default_rng()
    return x + std * rng.standard_normal(np.shape(x)).astype(np.asarray(x).dtype, copy=False)


def gaussian_sigma_for_epsilon(epsilon: float, delta: float, sensitivity: float = 1.0) -> float:
    """Classic single-release Gaussian mechanism (Dwork & Roth, Thm. A.1; needs epsilon < 1):
    ``sigma = sqrt(2 ln(1.25/delta)) * sensitivity / epsilon``."""
    if not 0 < epsilon < 1:
        raise ValueError("The classic Gaussian mechanism bound needs 0 < epsilon < 1; "
                         "use the RDP accountant for larger epsilon.")
    return float(np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon)


def effective_noise_multiplier(noise_multiplier: float, num_clients: int, num_colluders: int) -> float:
    """Distributed DP: ``n`` clients each add ``N(0, sigma^2 C^2 / n)`` so that the *sum*
    carries ``N(0, sigma^2 C^2)``. Colluders that subtract their own noise share leave only
    ``n - k`` honest shares: effective multiplier ``sigma * sqrt((n - k) / n)``."""
    honest = max(num_clients - num_colluders, 0)
    return float(noise_multiplier * np.sqrt(honest / num_clients))


def privatize_mean(uploads, max_norm: float, noise_multiplier: float, denominator: Optional[float] = None,
                   generator=None):
    """Central DP-FedAvg in one call: clip every row, sum, add ``N(0, (sigma C)^2)``, divide."""
    from ..backend import as_matrix

    X, xp = as_matrix(uploads)
    clipped = xp.stack([clip_l2(X[i], max_norm) for i in range(X.shape[0])])
    total = add_gaussian_noise(xp.sum(clipped, axis=0), noise_multiplier * max_norm, generator)
    return total / (denominator if denominator is not None else X.shape[0])
