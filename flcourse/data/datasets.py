"""



Disclaimer:

- Note that there are some unusual optimisations here to make things work even on non-GPU machines. The main decision I did was to keep the whole dataset in memory. Implementation of this class done by an AI
- Images are compressed in uint8 and normalised on the fly in :meth:`ArrayDataset.batch`,
which keeps CIFAR-10 at 150 MB of RAM instead of 600 MB.


``load_dataset(cfg.data)`` returns ``(train, test)`` as :class:`ArrayDataset` objects.

Supported datasets

==============  =====================  ==========================================
name            input shape            notes
==============  =====================  ==========================================
mnist           1 x 28 x 28            torchvision download (~11 MB)
fashion_mnist   1 x 28 x 28            torchvision download (~30 MB)
cifar10         3 x 32 x 32            torchvision download (~170 MB)
digits          1 x 8 x 8              bundled with scikit-learn -> works offline
synthetic       20 features            Gaussian mixture, no download (unit tests)
==============  =====================  ==========================================

"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence, Tuple

import numpy as np
import torch

# per-channel normalisation constants
_STATS = {
    "mnist": ((0.1307,), (0.3081,)),
    "fashion_mnist": ((0.2860,), (0.3530,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}


@dataclass
class ArrayDataset:
    """Inputs ``x`` (uint8 images ``(N, C, H, W)`` or float features ``(N, D)``) and labels ``y``."""

    x: torch.Tensor
    y: torch.Tensor
    num_classes: int
    mean: Optional[Sequence[float]] = None   # per-channel mean (images only)
    std: Optional[Sequence[float]] = None
    name: str = ""
    _mean_t: Optional[torch.Tensor] = field(default=None, repr=False)
    _std_t: Optional[torch.Tensor] = field(default=None, repr=False)

    def __post_init__(self):
        self.y = self.y.long()
        if self.mean is not None:
            shape = (1, -1, 1, 1)
            self._mean_t = torch.tensor(self.mean, dtype=torch.float32).view(shape)
            self._std_t = torch.tensor(self.std, dtype=torch.float32).view(shape)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return tuple(self.x.shape[1:])

    @property
    def device(self) -> torch.device:
        return self.x.device

    def to(self, device) -> "ArrayDataset":
        """Move the raw data to ``device`` (useful on a GPU: no host->device copies)."""
        out = ArrayDataset(self.x.to(device), self.y.to(device), self.num_classes,
                           self.mean, self.std, self.name)
        return out

    def subset(self, indices) -> "ArrayDataset":
        idx = torch.as_tensor(np.asarray(indices), dtype=torch.long, device=self.x.device)
        return ArrayDataset(self.x[idx], self.y[idx], self.num_classes, self.mean, self.std, self.name)

    def with_data(self, x: torch.Tensor, y: torch.Tensor) -> "ArrayDataset":
        """Same normalisation, different raw data (used by data-poisoning attacks)."""
        return ArrayDataset(x, y, self.num_classes, self.mean, self.std, self.name)

    def normalize(self, xb: torch.Tensor) -> torch.Tensor:
        if self._mean_t is None:
            return xb.float()
        mean = self._mean_t.to(xb.device)
        std = self._std_t.to(xb.device)
        return (xb.float() / 255.0 - mean) / std

    def batch(self, idx, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return normalised float inputs and labels for the given indices."""
        xb, yb = self.x[idx], self.y[idx]
        if device is not None:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        return self.normalize(xb), yb

    def batches(self, batch_size: int, shuffle: bool = True, generator: torch.Generator | None = None,
                device=None, drop_last: bool = False) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate once over the dataset in mini-batches."""
        n = len(self)
        order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
        order = order.to(self.x.device)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            if drop_last and len(idx) < batch_size:
                break
            yield self.batch(idx, device)

    def label_counts(self) -> np.ndarray:
        return np.bincount(self.y.cpu().numpy(), minlength=self.num_classes)


# ----------------------------------------------------------------------------------
# loaders
# ----------------------------------------------------------------------------------
def _torchvision_dataset(name: str, root: str):
    try:
        import torchvision
    except ImportError as exc:  # pragma: no cover - depends on the installation
        raise ImportError("torchvision is needed for mnist/fashion_mnist/cifar10. "
                          "`pip install torchvision`, or use dataset: digits / synthetic.") from exc
    cls = {"mnist": torchvision.datasets.MNIST,
           "fashion_mnist": torchvision.datasets.FashionMNIST,
           "cifar10": torchvision.datasets.CIFAR10}[name]
    out = []
    for train in (True, False):
        ds = cls(root=root, train=train, download=True)
        x = torch.as_tensor(np.asarray(ds.data))
        if x.ndim == 3:            # (N, H, W) -> (N, 1, H, W)
            x = x.unsqueeze(1)
        else:                      # CIFAR: (N, H, W, C) -> (N, C, H, W)
            x = x.permute(0, 3, 1, 2).contiguous()
        y = torch.as_tensor(np.asarray(ds.targets), dtype=torch.long)
        mean, std = _STATS[name]
        out.append(ArrayDataset(x.to(torch.uint8), y, 10, mean, std, name))
    return out[0], out[1]


def _digits(seed: int):
    try:
        from sklearn.datasets import load_digits
    except ImportError as exc:  # pragma: no cover
        raise ImportError("dataset 'digits' needs scikit-learn (`pip install scikit-learn`).") from exc
    d = load_digits()
    x = np.round(d.images * (255.0 / 16.0)).astype(np.uint8)[:, None, :, :]   # (N, 1, 8, 8)
    y = d.target.astype(np.int64)
    rng = np.random.default_rng(1234)          # fixed split, independent of the run seed
    order = rng.permutation(len(y))
    n_test = 360
    test_idx, train_idx = order[:n_test], order[n_test:]
    xtr = x[train_idx].astype(np.float64) / 255.0
    mean, std = (float(xtr.mean()),), (float(xtr.std()),)
    train = ArrayDataset(torch.as_tensor(x[train_idx]), torch.as_tensor(y[train_idx]), 10, mean, std, "digits")
    test = ArrayDataset(torch.as_tensor(x[test_idx]), torch.as_tensor(y[test_idx]), 10, mean, std, "digits")
    return train, test


def _synthetic(num_classes: int = 10, dim: int = 20, n_train: int = 6000, n_test: int = 1000,
               separation: float = 1.5):
    """Gaussian mixture classification problem (no download, deterministic)."""
    g = torch.Generator().manual_seed(1234)
    centers = torch.randn(num_classes, dim, generator=g) * separation

    def sample(n):
        y = torch.randint(0, num_classes, (n,), generator=g)
        x = centers[y] + torch.randn(n, dim, generator=g)
        return x.float(), y

    xtr, ytr = sample(n_train)
    xte, yte = sample(n_test)
    return (ArrayDataset(xtr, ytr, num_classes, name="synthetic"),
            ArrayDataset(xte, yte, num_classes, name="synthetic"))


def _subsample(ds: ArrayDataset, n: Optional[int], seed: int) -> ArrayDataset:
    if n is None or n >= len(ds):
        return ds
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(ds), size=int(n), replace=False))
    return ds.subset(idx)


def load_dataset(data_cfg, seed: int = 0) -> Tuple[ArrayDataset, ArrayDataset]:
    """Load the dataset described by the ``data`` section of a config."""
    name = str(data_cfg.get("dataset", "mnist")).lower().replace("-", "_")
    root = data_cfg.get("root", "./data")
    if name in ("mnist", "fashion_mnist", "fmnist", "cifar10", "cifar"):
        name = {"fmnist": "fashion_mnist", "cifar": "cifar10"}.get(name, name)
        train, test = _torchvision_dataset(name, root)
    elif name == "digits":
        train, test = _digits(seed)
    elif name == "synthetic":
        train, test = _synthetic(**(data_cfg.get("synthetic") or {}))
    else:
        raise ValueError(f"Unknown dataset {name!r}. Choose mnist, fashion_mnist, cifar10, digits or synthetic.")
    train = _subsample(train, data_cfg.get("max_train_samples"), seed)
    test = _subsample(test, data_cfg.get("max_test_samples"), seed + 1)
    return train, test
