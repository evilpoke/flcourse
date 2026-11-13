"""Small demo networks + helpers to flatten a model into one parameter vector.

In flcourse a client upload is always a single flat vector (``torch.Tensor`` of shape
``(d,)`` or a numpy array). :func:`get_flat_params` / :func:`set_flat_params` convert
between an ``nn.Module`` and such a vector.

=============  ===========================================  ====================
name           architecture                                 #params (MNIST)
=============  ===========================================  ====================
logreg         linear softmax regression                    7.9 k
mlp            2 hidden layers x 200 (McMahan et al. "2NN")  199 k
lenet5         LeNet-5 (activation relu or sigmoid)         61.7 k
cnn            2x conv3x3 (16, 32) + FC 64  (CPU friendly)  106 k
cnn_large      McMahan et al. CNN, 2x conv5x5 (32, 64)+512   1.66 M
resnet18       ResNet-18 for 32x32 inputs with GroupNorm     11.2 M (CIFAR)
dlg_lenet      sigmoid LeNet of the DLG paper (inversion)    13.4 k
=============  ===========================================  ====================

No BatchNorm: its running statistics are buffers, not parameters, and would not be
part of the uploaded vector. GroupNorm is the usual replacement in FL (Hsieh et al. 2020).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _act(name: str) -> nn.Module:
    return {"relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh()}[name]


class LogReg(nn.Module):
    def __init__(self, input_shape: Sequence[int], num_classes: int):
        super().__init__()
        self.fc = nn.Linear(int(np.prod(input_shape)), num_classes)

    def forward(self, x):
        return self.fc(x.flatten(1))


class MLP(nn.Module):
    def __init__(self, input_shape: Sequence[int], num_classes: int, hidden: int = 200,
                 activation: str = "relu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(np.prod(input_shape)), hidden), _act(activation),
            nn.Linear(hidden, hidden), _act(activation),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class LeNet5(nn.Module):
    def __init__(self, input_shape: Sequence[int], num_classes: int, activation: str = "relu"):
        super().__init__()
        c, h, w = input_shape
        if min(h, w) < 16:
            raise ValueError("lenet5 needs inputs of at least 16x16 pixels; use mlp for 'digits'.")
        self.features = nn.Sequential(
            nn.Conv2d(c, 6, 5, padding=2), _act(activation), nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5), _act(activation), nn.MaxPool2d(2),
        )
        with torch.no_grad():
            flat = self.features(torch.zeros(1, c, h, w)).numel()
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(flat, 120), _act(activation),
            nn.Linear(120, 84), _act(activation), nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class SmallCNN(nn.Module):
    def __init__(self, input_shape: Sequence[int], num_classes: int, channels=(16, 32), hidden: int = 64,
                 kernel: int = 3):
        super().__init__()
        c, h, w = input_shape
        c1, c2 = channels
        self.features = nn.Sequential(
            nn.Conv2d(c, c1, kernel, padding=kernel // 2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel, padding=kernel // 2), nn.ReLU(), nn.MaxPool2d(2),
        )
        with torch.no_grad():
            flat = self.features(torch.zeros(1, c, h, w)).numel()
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(flat, hidden), nn.ReLU(),
                                        nn.Linear(hidden, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


class DLGLeNet(nn.Module):
    """The LeNet variant of the DLG paper (Zhu et al. 2019): sigmoid activations, strided
    convolutions, no pooling, weights ~ U(-0.5, 0.5). Twice differentiable and easy to
    invert -- used by the gradient-inversion demo (experiments/privacy_attacks.py)."""

    def __init__(self, input_shape: Sequence[int], num_classes: int):
        super().__init__()
        c, h, w = input_shape
        self.body = nn.Sequential(
            nn.Conv2d(c, 12, 5, padding=2, stride=2), nn.Sigmoid(),
            nn.Conv2d(12, 12, 5, padding=2, stride=2), nn.Sigmoid(),
            nn.Conv2d(12, 12, 5, padding=2, stride=1), nn.Sigmoid(),
        )
        with torch.no_grad():
            flat = self.body(torch.zeros(1, c, h, w)).numel()
        self.fc = nn.Linear(flat, num_classes)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                m.weight.data.uniform_(-0.5, 0.5)
                m.bias.data.uniform_(-0.5, 0.5)

    def forward(self, x):
        return self.fc(self.body(x).flatten(1))


class _BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int, groups: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = nn.GroupNorm(groups, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = nn.GroupNorm(groups, cout)
        self.short = nn.Sequential()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.GroupNorm(groups, cout))

    def forward(self, x):
        out = F.relu(self.n1(self.conv1(x)))
        out = self.n2(self.conv2(out))
        return F.relu(out + self.short(x))


class ResNet18(nn.Module):
    """ResNet-18 for 32x32 images (3x3 stem, no max-pool) with GroupNorm."""

    def __init__(self, input_shape: Sequence[int], num_classes: int):
        super().__init__()
        c = input_shape[0]
        self.stem = nn.Sequential(nn.Conv2d(c, 64, 3, 1, 1, bias=False), nn.GroupNorm(2, 64), nn.ReLU())
        layers, cin = [], 64
        for cout, stride in [(64, 1), (128, 2), (256, 2), (512, 2)]:
            layers += [_BasicBlock(cin, cout, stride), _BasicBlock(cout, cout, 1)]
            cin = cout
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.layers(self.stem(x))
        return self.fc(F.adaptive_avg_pool2d(out, 1).flatten(1))


def build_model(model_cfg, input_shape: Sequence[int], num_classes: int) -> nn.Module:
    """Create the model described by the ``model`` section of a config."""
    if isinstance(model_cfg, str):
        model_cfg = {"name": model_cfg}
    name = model_cfg.get("name", "mlp").lower()
    kwargs = {k: v for k, v in model_cfg.items() if k != "name"}
    if name == "logreg":
        return LogReg(input_shape, num_classes)
    if name == "mlp":
        return MLP(input_shape, num_classes, **kwargs)
    if name in ("lenet", "lenet5"):
        return LeNet5(input_shape, num_classes, **kwargs)
    if name == "cnn":
        return SmallCNN(input_shape, num_classes, **kwargs)
    if name == "cnn_large":
        return SmallCNN(input_shape, num_classes, channels=(32, 64), hidden=512, kernel=5)
    if name == "resnet18":
        return ResNet18(input_shape, num_classes)
    if name == "dlg_lenet":
        return DLGLeNet(input_shape, num_classes)
    raise ValueError(f"Unknown model {name!r}: logreg, mlp, lenet5, cnn, cnn_large, resnet18, dlg_lenet.")


# ----------------------------------------------------------------------------------
# model <-> flat vector
# ----------------------------------------------------------------------------------
def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def get_flat_params(model: nn.Module) -> torch.Tensor:
    """All parameters concatenated into a fresh 1-D tensor (a copy, not a view)."""
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()


@torch.no_grad()
def set_flat_params(model: nn.Module, vector) -> None:
    """Copy a flat vector (torch or numpy) into the model parameters.

    Uses ``copy_`` on purpose: ``torch.nn.utils.vector_to_parameters`` makes the
    parameters *views* of the vector, so an optimizer step would silently modify the
    global model vector as well.
    """
    if not isinstance(vector, torch.Tensor):
        vector = torch.as_tensor(np.asarray(vector))
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.copy_(vector[offset:offset + n].view_as(p).to(device=p.device, dtype=p.dtype))
        offset += n
    if offset != vector.numel():
        raise ValueError(f"Vector has {vector.numel()} entries but the model has {offset} parameters.")


def get_flat_grad(model: nn.Module) -> torch.Tensor:
    """All ``.grad`` fields concatenated into a 1-D tensor (zeros for unused parameters)."""
    grads = []
    for p in model.parameters():
        g = p.grad if p.grad is not None else torch.zeros_like(p)
        grads.append(g.detach().reshape(-1))
    return torch.cat(grads).clone()
