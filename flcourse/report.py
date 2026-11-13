"""Figures for an experiment run (matplotlib, PNG files).

* :func:`plot_convergence`   -- per-client training loss + global test loss/accuracy
* :func:`plot_communication` -- bytes per protocol message type and per round
* :func:`plot_computation`   -- time per step: average over parties vs. parallel estimate (max)
* :func:`plot_partition`     -- label distribution of every client (IID vs non-IID)
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from .bench import Bench, fmt_bits  # noqa: E402

HONEST, MALICIOUS, GLOBAL = "#4C78A8", "#E45756", "#222222"


def _save(fig, path) -> Path:
    path = Path(path)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_convergence(global_rows: List[dict], client_rows: List[dict], path, title: str = "") -> Optional[Path]:
    """Three panels: (a) every client's local training loss, (b) global test loss,
    (c) global test accuracy (plus backdoor attack success rate if logged)."""
    if not global_rows and not client_rows:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    # (a) clients ---------------------------------------------------------------------
    ax = axes[0]
    per_client: Dict[int, List[tuple]] = defaultdict(list)
    malicious = {}
    for r in client_rows:
        if "loss" in r and r["loss"] is not None and np.isfinite(r["loss"]):
            per_client[r["client"]].append((r["round"], r["loss"]))
            malicious[r["client"]] = bool(r.get("malicious", False))
    for cid, pts in sorted(per_client.items()):
        pts.sort()
        xs, ys = zip(*pts)
        color = MALICIOUS if malicious[cid] else HONEST
        ax.plot(xs, ys, "-o", color=color, alpha=0.35, lw=1, ms=2.5)
    honest_by_round: Dict[int, List[float]] = defaultdict(list)
    for r in client_rows:
        if not r.get("malicious", False) and r.get("loss") is not None and np.isfinite(r["loss"]):
            honest_by_round[r["round"]].append(r["loss"])
    if honest_by_round:
        xs = sorted(honest_by_round)
        ax.plot(xs, [np.mean(honest_by_round[x]) for x in xs], color=GLOBAL, lw=2.2, label="mean (honest clients)")
    ax.plot([], [], color=HONEST, alpha=0.6, label="individual honest client")
    if any(malicious.values()):
        ax.plot([], [], color=MALICIOUS, alpha=0.6, label="malicious client")
    ax.set_title("(a) client local training loss")
    ax.set_xlabel("round")
    ax.set_ylabel("cross-entropy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) + (c) federator ---------------------------------------------------------------
    g = [r for r in global_rows if "test_loss" in r]
    if g:
        xs = [r["round"] for r in g]
        axes[1].plot(xs, [r["test_loss"] for r in g], "-o", color=GLOBAL, ms=3)
        axes[2].plot(xs, [r["test_acc"] for r in g], "-o", color=GLOBAL, ms=3, label="rate of correct classification")
        if any("backdoor_asr" in r for r in g):
            axes[2].plot(xs, [r.get("backdoor_asr", np.nan) for r in g], "--s", color=MALICIOUS, ms=3,
                         label="backdoor success rate")
        if any("epsilon" in r for r in g):
            ax2 = axes[2].twinx()
            ax2.plot(xs, [r.get("epsilon", np.nan) for r in g], ":", color="#54A24B", label="DP epsilon")
            ax2.set_ylabel("epsilon (DP)", color="#54A24B")
        axes[2].legend(fontsize=8, loc="lower right")
    axes[1].set_title("(b) global model: test loss")
    axes[1].set_xlabel("round")
    axes[1].set_ylabel("cross-entropy")
    axes[1].grid(alpha=0.3)
    axes[2].set_title("(c) global model: test accuracy")
    axes[2].set_xlabel("round")
    axes[2].set_ylim(0, 1)
    axes[2].grid(alpha=0.3)
    for a in axes:
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
    if title:
        fig.suptitle(title, fontsize=11)
    return _save(fig, path)


def plot_communication(bench: Bench, path, title: str = "") -> Optional[Path]:
    rows = bench.comm_summary()
    if not rows:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(15, max(3.2, 0.42 * len(rows) + 1.5)))
    colors = {"client->server": "#4C78A8", "server->client": "#F58518", "client->colluders": "#E45756",
              "server->server": "#999999"}
    labels = [f"{r['label']}  [{r['direction']}]" for r in rows]
    vals = [r["bits"] / 8 for r in rows]
    ax = axes[0]
    bars = ax.barh(range(len(rows)), vals, color=[colors.get(r["direction"], "#72B7B2") for r in rows])
    ax.set_yticks(range(len(rows)), labels, fontsize=8)
    ax.invert_yaxis()
    if max(vals) > 50 * max(min(vals), 1):
        ax.set_xscale("log")
        ax.set_xlabel("total bytes (log scale)")
    else:
        ax.set_xlabel("total bytes")
        ax.set_xlim(0, max(vals) * 1.25)
    ax.set_title("total communication per message type")
    for b, r in zip(bars, rows):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, " " + fmt_bits(r["bits"]), va="center", fontsize=7)
    ax.grid(alpha=0.3, axis="x")

    # per round, stacked by label
    ax = axes[1]
    per: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for rec in bench.comm:
        per[rec["label"]][rec["round"]] += rec["bits"] / 8 / 1e6
    rounds = sorted({rec["round"] for rec in bench.comm})
    bottom = np.zeros(len(rounds))
    cmap = plt.get_cmap("tab20")
    for i, (label, d) in enumerate(sorted(per.items())):
        v = np.array([d.get(rd, 0.0) for rd in rounds])
        ax.bar(rounds, v, bottom=bottom, label=label, color=cmap(i % 20))
        bottom += v
    ax.set_xlabel("round")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("MB")
    ax.set_title("communication per round")
    ax.legend(fontsize=7, ncol=1, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.grid(alpha=0.3, axis="y")
    if title:
        fig.suptitle(title, fontsize=11)
    return _save(fig, path)


def plot_computation(bench: Bench, path, title: str = "") -> Optional[Path]:
    rows = bench.time_summary()
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(10, max(3.0, 0.45 * len(rows) + 1.2)))
    y = np.arange(len(rows))
    ax.barh(y - 0.2, [r["avg_per_round_s"] for r in rows], height=0.4, color="#BAB0AC",
            label="average over parties")
    ax.barh(y + 0.2, [r["parallel_per_round_s"] for r in rows], height=0.4, color="#4C78A8",
            label="max over parties (concurrent clients)")
    ax.set_yticks(y, [f"{r['label']} [{r['who']}]" for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("seconds per round (log scale)")
    ax.set_title(title or "computation time per round")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    return _save(fig, path)


def plot_partition(counts: np.ndarray, path, title: str = "label distribution per client",
                   malicious: Optional[set] = None) -> Path:
    """Stacked bars: number of samples of every class held by each client."""
    num_clients, num_classes = counts.shape
    fig, ax = plt.subplots(figsize=(max(6, 0.35 * num_clients + 2), 3.6))
    bottom = np.zeros(num_clients)
    cmap = plt.get_cmap("tab10" if num_classes <= 10 else "tab20")
    for c in range(num_classes):
        ax.bar(range(num_clients), counts[:, c], bottom=bottom, color=cmap(c % cmap.N), label=str(c))
        bottom += counts[:, c]
    ticks = [f"{i}*" if malicious and i in malicious else str(i) for i in range(num_clients)]
    ax.set_xticks(range(num_clients), ticks, fontsize=7)
    ax.set_xlabel("client  (* = malicious)" if malicious else "client")
    ax.set_ylabel("#samples")
    ax.set_title(title)
    ax.legend(title="class", fontsize=7, ncol=2, bbox_to_anchor=(1.0, 1.0), loc="upper left")
    return _save(fig, path)


def replot(run_dir) -> None:
    """Re-draw convergence / communication / computation figures of a finished run from its
    ``metrics.json`` and ``bench.json`` (e.g. after changing the plotting code)."""
    import json

    run_dir = Path(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    bench = Bench()
    if (run_dir / "bench.json").exists():
        raw = json.loads((run_dir / "bench.json").read_text())
        bench.comm, bench.time = raw["comm"], raw["time"]
    title = metrics.get("summary", {}).get("name", "") or ""
    plot_convergence(metrics["global"], metrics["clients"], run_dir / "convergence.png", title)
    plot_communication(bench, run_dir / "communication.png", title)
    plot_computation(bench, run_dir / "computation.png", title)
