"""Privacy attack of an honest-but-curious server: gradient inversion.

Deep Leakage from Gradients (DLG, Zhu et al., NeurIPS 2019) with the label trick of
iDLG (Zhao et al. 2020). Given the gradient ``g`` a client uploaded (``upload: gradient``)
and the global model, the server optimises a dummy input ``x'`` (and label) such that
the gradient of ``x'`` matches ``g``:

    min_{x'}  || grad_w L(f_w(x'), y') - g ||^2

For a single training example this recovers the private image almost perfectly; it
degrades with larger batches, DP noise, and secure aggregation (the server only sees a
sum over many clients).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def split_like_params(model: nn.Module, flat: torch.Tensor) -> List[torch.Tensor]:
    """Split a flat gradient vector into tensors shaped like the model parameters."""
    out, offset = [], 0
    for p in model.parameters():
        n = p.numel()
        out.append(flat[offset:offset + n].view_as(p).to(p.device, p.dtype))
        offset += n
    return out


def infer_label_idlg(model: nn.Module, flat_grad: torch.Tensor) -> int:
    """iDLG: for one example and non-negative inputs to the last linear layer, only the row
    of the true class has a negative gradient sum."""
    last_weight_grad = None
    for p, g in zip(model.parameters(), split_like_params(model, flat_grad)):
        if p.ndim == 2:
            last_weight_grad = g
    return int(torch.argmin(last_weight_grad.sum(dim=1)))


def invert_gradient(model: nn.Module, flat_grad: torch.Tensor, input_shape: Tuple[int, ...], num_classes: int,
                    label: Optional[int | Sequence[int]] = None, batch_size: int = 1, iterations: int = 100,
                    lr: float = 1.0, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    """Reconstruct ``batch_size`` inputs from a gradient with L-BFGS.

    ``label=None`` optimises soft dummy labels as in DLG; an int (or a sequence of
    ``batch_size`` ints) fixes the label(s) as known. Returns ``(x_rec, labels, history)``
    with ``x_rec`` in the *normalised* input space of the model.
    """
    device = next(model.parameters()).device
    model.eval()
    target = [g.detach() for g in split_like_params(model, flat_grad.to(device))]
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn((batch_size, *input_shape), generator=gen).to(device).requires_grad_(True)
    if label is None:
        y_logits = torch.randn((batch_size, num_classes), generator=gen).to(device).requires_grad_(True)
        variables = [x, y_logits]
    else:
        # one known label for the whole batch (int), or one known label per dummy image (sequence)
        if isinstance(label, (int, np.integer)):
            y_fixed = torch.full((batch_size,), int(label), dtype=torch.long, device=device)
        else:
            y_fixed = torch.as_tensor(list(label), dtype=torch.long, device=device)
            if y_fixed.numel() != batch_size:
                raise ValueError(f"got {y_fixed.numel()} labels for batch_size={batch_size}")
        variables = [x]
    opt = torch.optim.LBFGS(variables, lr=lr, max_iter=20, history_size=100, line_search_fn="strong_wolfe")
    history: List[float] = []

    def closure():
        opt.zero_grad()
        out = model(x)
        if label is None:
            loss = torch.mean(torch.sum(-F.softmax(y_logits, dim=-1) * F.log_softmax(out, dim=-1), dim=-1))
        else:
            loss = F.cross_entropy(out, y_fixed)
        grads = torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
        diff = sum(((g - t) ** 2).sum() for g, t in zip(grads, target))
        diff.backward()
        return diff

    for _ in range(iterations):
        value = float(opt.step(closure).detach())
        history.append(value)
        if value < 1e-10:
            break
    labels = y_logits.detach().argmax(dim=1) if label is None else y_fixed
    return x.detach(), labels, history
