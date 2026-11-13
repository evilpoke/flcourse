"""Server-side helpers: client sampling and the global model update."""
from __future__ import annotations

from typing import List

import numpy as np
import torch

from .backend import to_torch


def num_selected(num_clients: int, clients_per_round) -> int:
    """``clients_per_round`` may be an int or a fraction in (0, 1]."""
    if isinstance(clients_per_round, float) and clients_per_round <= 1.0:
        k = max(1, int(round(clients_per_round * num_clients)))
    else:
        k = int(clients_per_round)
    if not 1 <= k <= num_clients:
        raise ValueError(f"clients_per_round={clients_per_round} invalid for {num_clients} clients")
    return k


def sample_clients(num_clients: int, clients_per_round, rng: np.random.Generator) -> List[int]:
    """Uniformly sample a cohort of clients without replacement (sorted ids)."""
    k = num_selected(num_clients, clients_per_round)
    return sorted(rng.choice(num_clients, size=k, replace=False).tolist())


def apply_aggregate(global_vector: torch.Tensor, aggregate, upload: str, server_lr: float = 1.0) -> torch.Tensor:
    """Turn the aggregated upload into the next global model.

    * ``model``    : ``w <- aggregate``
    * ``delta``    : ``w <- w + server_lr * aggregate``
    * ``gradient`` : ``w <- w - server_lr * aggregate``
    """
    agg = to_torch(aggregate, like=global_vector)
    if upload == "model":
        return agg.clone()
    if upload == "delta":
        return global_vector + server_lr * agg
    if upload == "gradient":
        return global_vector - server_lr * agg
    raise ValueError(f"Unknown upload mode {upload!r}")
