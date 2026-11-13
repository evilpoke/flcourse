"""A federated client.

Clients are simulated *one after another* for an FL communication round.
We assume in real world deployment they would run concurrently in a deployment.
However, there is a "parallel estimate" in the benchmark report). 


An FL client can either submit the local model, the model difference or the gradient. 
All three methods kinda amount to the same for simple FedAvg aggregation but note that
for different aggregation, this is different!

In particular, we use ``model`` and ``delta`` for plain mean (FedAvg), we use ``delta`` whenever
an operation needs the *direction* of a client's change (DP clipping, FLTrust, centered
clipping, norm bounding, sign flipping, ...) and ``gradient`` for batched gradient descient like FedSGD.


"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .data.datasets import ArrayDataset
from .models import get_flat_params, set_flat_params
from .training import compute_gradient, local_sgd

UPLOAD_MODES = ("model", "delta", "gradient")


class Client:
    def __init__(self, cid: int, data: ArrayDataset, client_cfg, seed: int = 0, device=None,
                 malicious: bool = False):
        self.cid = cid
        self.data = data
        self.cfg = client_cfg
        self.device = device
        self.malicious = malicious
        # private RNG for the mini-batch order -> reproducible, independent of other clients
        self.generator = torch.Generator().manual_seed(10_007 * (seed + 1) + cid)

    @property
    def num_samples(self) -> int:
        return len(self.data)

    def local_update(self, model: nn.Module, global_vector: torch.Tensor,
                     upload: str = "model") -> Tuple[torch.Tensor, dict]:
        """Start from the global model and return ``(upload_vector, stats)``.

        ``stats`` contains the local training loss/accuracy (averaged over local steps),
        which is what the convergence plots show per client.
        """
        if upload not in UPLOAD_MODES:
            raise ValueError(f"fl.upload must be one of {UPLOAD_MODES}, got {upload!r}")
        set_flat_params(model, global_vector)
        c = self.cfg

        if upload == "gradient":
            grad, stats = compute_gradient(model, self.data, batch_size=c.get("batch_size", 32),
                                           num_batches=c.get("gradient_batches", 1),
                                           generator=self.generator, device=self.device)
            return grad, stats

        # either "model" or the delta of the model
        stats = local_sgd(model, self.data, epochs=c.get("local_epochs", 1), steps=c.get("local_steps"),
                          batch_size=c.get("batch_size", 32), lr=c.get("lr", 0.05),
                          momentum=c.get("momentum", 0.0), weight_decay=c.get("weight_decay", 0.0),
                          generator=self.generator, device=self.device)
        local = get_flat_params(model)

        if upload == "model":
            return local, stats
        else:
            # "delta"
            return local - global_vector.to(local.device), stats      # upload == "delta"

    def __repr__(self) -> str:
        tag = ", malicious" if self.malicious else ""
        return f"Client({self.cid}, n={self.num_samples}{tag})"


def make_clients(train: ArrayDataset, parts, client_cfg, seed: int = 0, device=None,
                 malicious_ids: Optional[set] = None):
    """Generates a collection of :class:`Client`.   
        The parameter ``parts`` specifies which index-selection from ``train`` are to be given to what client."""
    malicious_ids = malicious_ids or set()
    return [Client(cid, train.subset(idx), client_cfg, seed=seed, device=device,
                   malicious=cid in malicious_ids) for cid, idx in enumerate(parts)]
