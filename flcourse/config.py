"""YAML configuration handling.

A configuration is a nested dictionary with attribute access:

    cfg = load_config("experiments/configs/fedavg_mnist_iid.yaml",
                      overrides=["fl.rounds=3", "data.partition=dirichlet"])
    cfg.fl.rounds          # -> 3
    cfg["fl"]["rounds"]    # -> 3 (it is still a dict)

Every key that is not given in the YAML file is taken from ``DEFAULTS`` below, so a
config file only has to list what differs from the defaults. The complete, merged
configuration is written to the output folder of every run (``config.yaml``).
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

# This is the format expected in the config yaml files (look at the examples)

DEFAULTS: dict = {
    "name": "experiment",
    "seed": 0,
    "device": "auto",            # auto | cpu | cuda | mps   (auto falls back to cpu)
    "output_dir": "results",     # results/<name>/<timestamp>/
    "backend": "torch",          # library of the uploaded vectors: torch | numpy
    "verbose": True,
    "data": {
        "dataset": "mnist",      # mnist | fashion_mnist | cifar10 | digits | synthetic
        "root": "./data",        # download folder
        "partition": "iid",      # iid | dirichlet | shards
        "alpha": 0.5,            # Dirichlet concentration (small = very non-iid)
        "shards_per_client": 2,  # "shards": number of label shards per client
        "min_samples_per_client": 10,
        "max_train_samples": None,  # subsample the training set (quick runs)
        "max_test_samples": None,
    },
    "model": {"name": "mlp"},    # logreg | mlp | lenet5 | cnn | resnet18
    "fl": {
        "num_clients": 10,
        "clients_per_round": 10,  # int, or a float in (0, 1] = fraction of clients
        "rounds": 10,
        "upload": "model",       # model | delta | gradient   (what a client submits)
        "server_lr": 1.0,        # used for upload = delta / gradient
    },
    "client": {
        "local_epochs": 1,       # local training (upload = model / delta)
        "local_steps": None,     # if set: exact number of local SGD steps instead of epochs
        "batch_size": 32,
        "lr": 0.05,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "gradient_batches": 1,   # upload = gradient: average gradient over this many batches
    },
    "aggregator": {"name": "mean"},
    "pre_aggregators": [],
    "attack": None,
    "dp": None,
    "secagg": None,
    "eval": {
        "every": 1,              # evaluate the global model every k rounds
        "batch_size": 1000,
    },
    "network": {                 # only used for the *estimated* communication time
        "uplink_mbps": 10.0,
        "downlink_mbps": 50.0,
    },
}

"""
    This is just a boilerplate config handler with obvious I/O stuff. 
    Nothing interesting to see from here on out. (The interesting stuff is the config YAML above)

"""
class Config(dict):
    """A ``dict`` whose keys can also be read as attributes (``cfg.fl.rounds``)."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(f"Config has no key {key!r}. Available: {list(self)}") from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __deepcopy__(self, memo):
        return Config({k: copy.deepcopy(v, memo) for k, v in self.items()})

    def get_path(self, path: str, default: Any = None) -> Any:
        """``cfg.get_path("fl.rounds")`` -- returns ``default`` if a key is missing."""
        node: Any = self
        for part in path.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def to_dict(self) -> dict:
        return _to_plain(self)


def _to_config(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return Config({k: _to_config(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_config(v) for v in obj]
    return obj


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    return obj


def deep_merge(base: Mapping, update: Mapping) -> dict:
    """Recursively merge ``update`` into a copy of ``base``."""
    out = copy.deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _parse_override(item: str) -> dict:
    """``"fl.rounds=3"`` -> ``{"fl": {"rounds": 3}}`` (values are parsed as YAML)."""
    if "=" not in item:
        raise ValueError(f"Override {item!r} must look like key.subkey=value")
    path, raw = item.split("=", 1)
    value = yaml.safe_load(raw)
    node: dict = {}
    root = node
    parts = path.strip().split(".")
    for part in parts[:-1]:
        node[part] = {}
        node = node[part]
    node[parts[-1]] = value
    return root


def load_config(config: str | Path | Mapping | None = None,
                overrides: Iterable[str] | Mapping | None = None) -> Config:
    """Load a YAML file (or dict), merge it into ``DEFAULTS`` and apply ``overrides``.

    ``overrides`` is either a dict (merged recursively) or a list of strings of the form
    ``"section.key=value"`` -- the same syntax as the ``--set`` command line option.
    """
    if config is None:
        user: Mapping = {}
    elif isinstance(config, Mapping):
        user = config
    else:
        with open(config, "r", encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
    merged = deep_merge(DEFAULTS, user)
    if overrides:
        if isinstance(overrides, Mapping):
            merged = deep_merge(merged, overrides)
        else:
            for item in overrides:
                merged = deep_merge(merged, _parse_override(item))
    return _to_config(merged)


def save_config(cfg: Mapping, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(_to_plain(cfg), fh, sort_keys=False)


def parse_args(description: str = "flcourse experiment", argv=None) -> dict:
    """Command line interface shared by all experiment scripts.

    ``python experiments/fedavg.py --config cfg.yaml --set fl.rounds=3 device=cpu``
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", "-c", required=True, help="path to a YAML config file")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config entries, e.g. --set fl.rounds=5 seed=1")
    args = parser.parse_args(argv)
    return {"config": args.config, "overrides": args.set}
