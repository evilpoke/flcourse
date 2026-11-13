"""Plain PyTorch building blocks used by clients and the server.

* :func:`local_sgd`        -- local training for ``upload = model / delta``
* :func:`compute_gradient` -- one (or a few) mini-batch gradients for ``upload = gradient``
* :func:`evaluate`         -- loss / accuracy of a model on a dataset

They are intentionally short and free of framework magic so that students can modify
them (e.g. add DP-SGD, proximal terms (FedProx), data augmentation, ...).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data.datasets import ArrayDataset
from .models import get_flat_grad


def local_sgd(model: nn.Module, data: ArrayDataset, *, epochs: int = 1, steps: Optional[int] = None,
              batch_size: int = 32, lr: float = 0.05, momentum: float = 0.0, weight_decay: float = 0.0,
              generator: Optional[torch.Generator] = None, device=None) -> dict:
    """Run local SGD on ``data``. Trains ``model`` in place.

    Either ``epochs`` full passes over the data or exactly ``steps`` mini-batch steps
    (if ``steps`` is given). Returns ``{"loss", "acc", "steps"}`` averaged over all steps.
    """
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    total_loss, total_correct, total_seen, done = 0.0, 0, 0, 0
    max_epochs = epochs if steps is None else 10**9
    for _ in range(max_epochs):
        for xb, yb in data.batches(batch_size, shuffle=True, generator=generator, device=device):
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach()) * len(yb)
            total_correct += int((logits.detach().argmax(1) == yb).sum())
            total_seen += len(yb)
            done += 1
            if steps is not None and done >= steps:
                break
        if steps is not None and done >= steps:
            break
    return {"loss": total_loss / max(total_seen, 1), "acc": total_correct / max(total_seen, 1), "steps": done}


def compute_gradient(model: nn.Module, data: ArrayDataset, *, batch_size: int = 32, num_batches: int = 1,
                     generator: Optional[torch.Generator] = None, device=None) -> Tuple[torch.Tensor, dict]:
    """Average gradient of the loss over ``num_batches`` random mini-batches (FedSGD).

    ``num_batches = -1`` uses the full local dataset. The model is *not* updated.
    """
    model.train()
    model.zero_grad(set_to_none=True)
    total_loss, total_correct, total_seen, used = 0.0, 0, 0, 0
    for xb, yb in data.batches(batch_size, shuffle=True, generator=generator, device=device):
        logits = model(xb)
        # sum (not mean) so that the final gradient is the average over *all* samples
        loss = F.cross_entropy(logits, yb, reduction="sum")
        loss.backward()
        total_loss += float(loss.detach())
        total_correct += int((logits.detach().argmax(1) == yb).sum())
        total_seen += len(yb)
        used += 1
        if num_batches != -1 and used >= num_batches:
            break
    grad = get_flat_grad(model) / max(total_seen, 1)
    model.zero_grad(set_to_none=True)
    return grad, {"loss": total_loss / max(total_seen, 1), "acc": total_correct / max(total_seen, 1),
                  "steps": used}


@torch.no_grad()
def evaluate(model: nn.Module, data: ArrayDataset, batch_size: int = 1000, device=None) -> Tuple[float, float]:
    """Return ``(average cross-entropy loss, accuracy)`` of ``model`` on ``data``."""
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for xb, yb in data.batches(batch_size, shuffle=False, device=device):
        logits = model(xb)
        total_loss += float(F.cross_entropy(logits, yb, reduction="sum"))
        correct += int((logits.argmax(1) == yb).sum())
        seen += len(yb)
    return total_loss / max(seen, 1), correct / max(seen, 1)
