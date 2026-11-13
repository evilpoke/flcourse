"""Split a dataset among clients: IID and non-IID partitions.

All functions return a list with one index array per client.

* ``iid``        -- uniformly random, (almost) equal sizes.
* ``dirichlet``  -- label skew: for every class ``c`` the class samples are distributed
  with proportions ``p_c ~ Dir(alpha, ..., alpha)``. Small ``alpha`` (0.1) = very
  non-iid, large ``alpha`` (100) = close to iid. (Hsu et al. 2019; NIID-Bench, Li et al. 2022)
* ``shards``     -- pathological non-iid from McMahan et al. 2017: sort by label, cut into
  ``num_clients * shards_per_client`` shards, every client gets ``shards_per_client``
  shards, i.e. (typically) only that many classes.
"""
from __future__ import annotations

from typing import List

import numpy as np


def iid_partition(num_samples: int, num_clients: int, rng: np.random.Generator) -> List[np.ndarray]:
    order = rng.permutation(num_samples)
    return [np.sort(part) for part in np.array_split(order, num_clients)]


def dirichlet_partition(labels: np.ndarray, num_clients: int, alpha: float, rng: np.random.Generator,
                        min_size: int = 10, max_tries: int = 1000) -> List[np.ndarray]:
    """Label-skewed partition with Dirichlet(alpha) class proportions per client."""
    labels = np.asarray(labels)
    n, classes = len(labels), np.unique(labels)
    if min_size * num_clients > n:
        raise ValueError(f"Cannot give {num_clients} clients >= {min_size} samples each from {n} samples.")
    for _ in range(max_tries):
        parts: List[list] = [[] for _ in range(num_clients)]
        for c in classes:
            idx_c = rng.permutation(np.flatnonzero(labels == c))
            p = rng.dirichlet(np.full(num_clients, alpha))
            # balance trick of NIID-Bench: clients that already hold >= n/K samples get no more
            sizes = np.array([len(part) for part in parts])
            p = p * (sizes < n / num_clients)
            if p.sum() == 0:
                p = np.full(num_clients, 1.0 / num_clients)
            p = p / p.sum()
            cuts = (np.cumsum(p) * len(idx_c)).astype(int)[:-1]
            for part, chunk in zip(parts, np.split(idx_c, cuts)):
                part.extend(chunk.tolist())
        if min(len(part) for part in parts) >= min_size:
            return [np.sort(np.asarray(part, dtype=np.int64)) for part in parts]
    raise RuntimeError(f"Could not find a Dirichlet(alpha={alpha}) partition where every client has "
                       f">= {min_size} samples; increase alpha or lower data.min_samples_per_client.")


def shard_partition(labels: np.ndarray, num_clients: int, shards_per_client: int,
                    rng: np.random.Generator) -> List[np.ndarray]:
    """Pathological non-iid split (McMahan et al. 2017)."""
    labels = np.asarray(labels)
    num_shards = num_clients * shards_per_client
    order = np.argsort(labels, kind="stable")
    shards = np.array_split(order, num_shards)
    shard_ids = rng.permutation(num_shards)
    parts = []
    for k in range(num_clients):
        mine = shard_ids[k * shards_per_client:(k + 1) * shards_per_client]
        parts.append(np.sort(np.concatenate([shards[s] for s in mine])))
    return parts


def partition(labels, num_clients: int, scheme: str = "iid", alpha: float = 0.5,
              shards_per_client: int = 2, min_size: int = 10, seed: int = 0) -> List[np.ndarray]:
    """Dispatch to one of the partition schemes above."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    scheme = scheme.lower()
    if scheme == "iid":
        return iid_partition(len(labels), num_clients, rng)
    if scheme == "dirichlet":
        return dirichlet_partition(labels, num_clients, alpha, rng, min_size=min_size)
    if scheme in ("shards", "shard", "pathological"):
        return shard_partition(labels, num_clients, shards_per_client, rng)
    raise ValueError(f"Unknown partition scheme {scheme!r}: use iid, dirichlet or shards.")


def label_distribution(labels, parts: List[np.ndarray], num_classes: int) -> np.ndarray:
    """``(num_clients, num_classes)`` matrix of label counts per client."""
    labels = np.asarray(labels)
    return np.stack([np.bincount(labels[idx], minlength=num_classes) for idx in parts])
