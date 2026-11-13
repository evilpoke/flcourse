"""Experiment bookkeeping: output folder, seeding, device, metrics, report.

    run = Run(cfg)                      # creates results/<name>/<timestamp>/, activates a Bench
    fed = build_federation(cfg, run)    # data, partition, clients, model
    for rnd in run.rounds():            # 1..R (also tells the Bench the current round)
        ...
        run.log_client(rnd, cid, loss=..., acc=...)
        run.log_global(rnd, test_loss=..., test_acc=...)
    summary = run.finish()              # metrics.json, clients.csv, report.txt, *.png
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Set

import numpy as np
import torch
import torch.nn as nn

from .bench import Bench, fmt_bits, fmt_seconds
from .client import Client, make_clients
from .config import Config, save_config
from .data import ArrayDataset, label_distribution, load_dataset, partition
from .device import describe_device, get_device, set_seed
from .models import build_model, get_flat_params, num_params
from .report import plot_communication, plot_computation, plot_convergence, plot_partition


class Run:
    def __init__(self, cfg: Config, quiet: Optional[bool] = None):
        self.cfg = cfg
        self.verbose = cfg.get("verbose", True) if quiet is None else not quiet
        set_seed(int(cfg.seed))
        self.device = get_device(cfg.get("device", "auto"))
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out = Path(cfg.get("output_dir", "results")) / cfg.get("name", "experiment") / stamp
        k = 1
        while out.exists():
            k += 1
            out = out.with_name(f"{stamp}-{k}")
        out.mkdir(parents=True)
        self.out_dir = out
        save_config(cfg, out / "config.yaml")
        self.bench = Bench(cfg.get("name", "")).activate()
        self.global_rows: List[dict] = []
        self.client_rows: List[dict] = []
        self.summary: dict = {}
        self._t0 = time.time()
        self._t_round = time.time()
        self.log(f"== {cfg.get('name')} ==  device: {describe_device(self.device)}  ->  {self.out_dir}")

    # -- logging -----------------------------------------------------------------------
    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def rounds(self, num_rounds: Optional[int] = None) -> Iterator[int]:
        num_rounds = num_rounds or int(self.cfg.fl.rounds)
        for rnd in range(1, num_rounds + 1):
            self.bench.set_round(rnd)
            self._t_round = time.time()
            yield rnd

    def log_client(self, rnd: int, cid: int, **metrics) -> None:
        self.client_rows.append({"round": rnd, "client": int(cid), **_floats(metrics)})

    def log_global(self, rnd: int, **metrics) -> None:
        row = {"round": rnd, **_floats(metrics), "elapsed_s": time.time() - self._t0}
        self.global_rows.append(row)
        if self.verbose:
            clients = [r["loss"] for r in self.client_rows if r["round"] == rnd and "loss" in r
                       and not r.get("malicious", False)]
            parts = [f"round {rnd:>3}/{self.cfg.fl.rounds}"]
            if "test_acc" in row:
                parts.append(f"test acc {row['test_acc']:.4f}  loss {row['test_loss']:.4f}")
            if clients:
                parts.append(f"client loss {np.mean(clients):.4f}")
            for key in ("backdoor_asr", "epsilon"):
                if key in row:
                    parts.append(f"{key} {row[key]:.3f}")
            parts.append(f"{time.time() - self._t_round:.1f}s")
            print(" | ".join(parts), flush=True)

    # -- end of experiment -------------------------------------------------------------
    def finish(self, title: Optional[str] = None, notes: str = "", **extra) -> dict:
        cfg = self.cfg
        title = title or cfg.get("name", "")
        net = cfg.get("network", {}) or {}
        up, down = float(net.get("uplink_mbps", 10.0)), float(net.get("downlink_mbps", 50.0))

        figures = {
            "convergence": plot_convergence(self.global_rows, self.client_rows, self.out_dir / "convergence.png", title),
            "communication": plot_communication(self.bench, self.out_dir / "communication.png", title),
            "computation": plot_computation(self.bench, self.out_dir / "computation.png", title),
        }
        final = self.global_rows[-1] if self.global_rows else {}
        latency = self.bench.latency_estimate(up, down)
        total_bits = sum(r["bits"] for r in self.bench.comm)
        self.summary.update({
            "name": cfg.get("name"), "output_dir": str(self.out_dir), "device": str(self.device),
            "final": final, "total_communication_bits": total_bits,
            "wall_clock_s": time.time() - self._t0, "estimated_round_latency": latency, **extra,
        })

        with open(self.out_dir / "metrics.json", "w") as fh:
            json.dump({"summary": self.summary, "global": self.global_rows, "clients": self.client_rows},
                      fh, indent=1, default=_json_default)
        with open(self.out_dir / "bench.json", "w") as fh:
            json.dump(self.bench.to_dict(), fh, default=_json_default)
        if self.client_rows:
            keys = sorted({k for r in self.client_rows for k in r}, key=lambda k: (k not in ("round", "client"), k))
            with open(self.out_dir / "clients.csv", "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.client_rows)

        lines = [f"EXPERIMENT  {title}", f"output      {self.out_dir}", f"device      {describe_device(self.device)}",
                 f"wall clock  {fmt_seconds(self.summary['wall_clock_s'])} (sequential simulation)", ""]
        if final:
            lines.append("FINAL METRICS (last evaluated round)")
            lines += [f"  {k:<16} {v:.4f}" if isinstance(v, float) else f"  {k:<16} {v}"
                      for k, v in final.items()]
            lines.append("")
        for k, v in extra.items():
            lines.append(f"{k}: {v}")
        if extra:
            lines.append("")
        if notes:
            lines += [notes, ""]
        lines.append(self.bench.report(up, down))
        lines.append("")
        lines.append("figures: " + ", ".join(p.name for p in figures.values() if p is not None))
        text = "\n".join(lines)
        (self.out_dir / "report.txt").write_text(text + "\n")
        self.log("\n" + text)
        self.bench.deactivate()
        return self.summary


def _floats(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            v = v.item()
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        out[k] = v
    return out


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


# ----------------------------------------------------------------------------------
# building the federation
# ----------------------------------------------------------------------------------
@dataclass
class Federation:
    train: ArrayDataset
    test: ArrayDataset
    parts: List[np.ndarray]
    clients: List[Client]
    model: nn.Module              # one shared "worker" network (clients are simulated sequentially)
    global_vector: torch.Tensor   # the initial global model as a flat vector
    malicious: Set[int]

    @property
    def num_params(self) -> int:
        return int(self.global_vector.numel())


def pick_malicious(num_clients: int, num_malicious: int, seed: int) -> Set[int]:
    """Fixed random set of malicious client ids (reproducible)."""
    rng = np.random.default_rng(seed + 99)
    return set(rng.choice(num_clients, size=num_malicious, replace=False).tolist()) if num_malicious else set()


def build_federation(cfg: Config, run: Run, malicious: Optional[Set[int]] = None) -> Federation:
    """Load + partition the data, create clients and the model."""
    t0 = time.time()
    train, test = load_dataset(cfg.data, seed=int(cfg.seed))
    if run.device.type != "cpu":           # keep the (uint8) data on the GPU: no copies per batch
        train, test = train.to(run.device), test.to(run.device)
    d = cfg.data
    parts = partition(train.y.cpu().numpy(), int(cfg.fl.num_clients), d.get("partition", "iid"),
                      alpha=float(d.get("alpha", 0.5)), shards_per_client=int(d.get("shards_per_client", 2)),
                      min_size=int(d.get("min_samples_per_client", 10)), seed=int(cfg.seed))
    malicious = set(malicious or set())
    clients = make_clients(train, parts, cfg.client, seed=int(cfg.seed), device=run.device, malicious_ids=malicious)
    model = build_model(cfg.model, train.input_shape, train.num_classes).to(run.device)
    global_vector = get_flat_params(model)

    counts = label_distribution(train.y.cpu().numpy(), parts, train.num_classes)
    np.savetxt(run.out_dir / "partition_counts.csv", counts, fmt="%d", delimiter=",")
    plot_partition(counts, run.out_dir / "partition.png",
                   title=f"{train.name}: {d.get('partition')} partition over {len(parts)} clients", malicious=malicious)
    sizes = [len(p) for p in parts]
    run.log(f"data: {train.name}  train={len(train)} test={len(test)}  partition={d.get('partition')}  "
            f"samples/client min={min(sizes)} max={max(sizes)}")
    run.log(f"model: {cfg.model.get('name')}  {num_params(model):,} parameters "
            f"({fmt_bits(32 * num_params(model))} as float32)  setup {time.time() - t0:.1f}s")
    if malicious:
        run.log(f"malicious clients: {sorted(malicious)}")
    return Federation(train, test, parts, clients, model, global_vector, malicious)
