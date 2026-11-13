"""Data-poisoning attacks: they change the *local data* of malicious clients.

* :func:`flip_labels`        -- untargeted label flipping ``y -> C - 1 - y``
* :func:`targeted_flip`      -- relabel one source class as a target class
* :func:`add_trigger`        -- stamp a backdoor trigger (white square / feature pattern)
* :func:`poison_backdoor`    -- BadNets-style backdoor on a fraction of the local samples
* :func:`backdoor_testset`   -- triggered test set to measure the attack success rate (ASR)
"""
from __future__ import annotations

import numpy as np
import torch

from ..data.datasets import ArrayDataset


def flip_labels(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    return (num_classes - 1) - y


def targeted_flip(y: torch.Tensor, source: int, target: int) -> torch.Tensor:
    y = y.clone()
    y[y == source] = target
    return y


def label_flipping(data: ArrayDataset, source: int | None = None, target: int | None = None) -> ArrayDataset:
    """Untargeted (default) or targeted (``source -> target``) label flipping."""
    y = flip_labels(data.y, data.num_classes) if source is None else targeted_flip(data.y, source, target)
    return data.with_data(data.x, y)


def add_trigger(x: torch.Tensor, size: int = 3, value: float | None = None) -> torch.Tensor:
    """Return a copy of ``x`` with a trigger.

    * images ``(N, C, H, W)``: a ``size x size`` square of maximal intensity in the
      bottom-right corner (BadNets, Gu et al. 2017);
    * feature vectors ``(N, D)``: the last ``size`` features are set to ``value`` (default 5).
    """
    x = x.clone()
    if x.ndim == 4:
        v = value if value is not None else (255 if x.dtype == torch.uint8 else float(x.max()))
        x[:, :, -size - 1:-1, -size - 1:-1] = v
    elif x.ndim == 2:
        x[:, -size:] = 5.0 if value is None else value
    else:
        raise ValueError(f"Unsupported input shape {tuple(x.shape)} for a trigger.")
    return x


def poison_backdoor(data: ArrayDataset, target_label: int = 0, fraction: float = 0.5, size: int = 3,
                    seed: int = 0) -> ArrayDataset:
    """Replace ``fraction`` of the local samples by triggered copies labelled ``target_label``."""
    n = len(data)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=int(round(fraction * n)), replace=False)
    idx_t = torch.as_tensor(idx, dtype=torch.long, device=data.x.device)
    x, y = data.x.clone(), data.y.clone()
    x[idx_t] = add_trigger(x[idx_t], size=size)
    y[idx_t] = target_label
    return data.with_data(x, y)


def backdoor_testset(test: ArrayDataset, target_label: int = 0, size: int = 3) -> ArrayDataset:
    """All test samples whose true label is not the target, triggered and labelled ``target``.
    The accuracy of a model on this set is the backdoor *attack success rate*."""
    keep = (test.y != target_label).nonzero().flatten()
    x = add_trigger(test.x[keep], size=size)
    y = torch.full_like(test.y[keep], target_label)
    return test.with_data(x, y)
