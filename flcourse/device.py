"""Robust device selection and seeding.

``get_device("auto")`` returns CUDA if a *working* GPU is present, then Apple's MPS, and
otherwise the CPU. "Working" means that a tiny tensor can actually be allocated and
used -- ``torch.cuda.is_available()`` alone can be ``True`` on machines with broken
drivers. Any failure silently falls back to the CPU (with a warning), so every
experiment runs on a laptop without a graphics card.
"""
from __future__ import annotations

import os
import random
import warnings

import numpy as np
import torch


def _works(device: str) -> bool:
    try:
        x = torch.ones(8, device=device)
        return float((x * 2).sum().cpu()) == 16.0
    except Exception:  # noqa: BLE001 - any failure means "not usable"
        return False


def get_device(preference: str = "auto") -> torch.device:
    """Pick a torch device: ``"auto"`` (default), ``"cpu"``, ``"cuda"``, ``"cuda:1"``, ``"mps"``."""
    preference = (preference or "auto").lower()
    if preference == "cpu":
        return torch.device("cpu")

    candidates = []
    if preference == "auto":
        candidates = ["cuda", "mps"]
    elif preference.startswith("cuda") or preference == "mps":
        candidates = [preference]
    else:
        warnings.warn(f"Unknown device {preference!r}; using the CPU.")

    for name in candidates:
        try:
            if name.startswith("cuda"):
                available = torch.cuda.is_available()
            else:
                available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        except Exception:  # noqa: BLE001
            available = False
        if available and _works(name):
            return torch.device(name)
        if preference != "auto":
            warnings.warn(f"Device {name!r} requested but not usable; falling back to the CPU.")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU and all GPUs)."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed)
        except Exception:  # noqa: BLE001
            pass
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        try:
            return f"cuda ({torch.cuda.get_device_name(device)})"
        except Exception:  # noqa: BLE001
            return "cuda"
    if device.type == "mps":
        return "mps (Apple GPU)"
    return f"cpu ({torch.get_num_threads()} threads)"
