"""Micro-benchmark: what does secure aggregation cost as the number of clients grows?

    python experiments/secagg_scaling.py --config experiments/configs/secagg_scaling.yaml

No training: every client holds a random vector of dimension ``dim`` and one aggregation
is run per (protocol, number of clients). For every run the communication of a client
(all key material, secret shares and masked vectors) and the computation of clients and
server are recorded with ``measure`` / ``timer`` into a separate ``Bench``.

Output: scaling.png (4 panels), scaling.csv and the table in report.txt.
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from flcourse.bench import Bench, fmt_bits, fmt_seconds, kind, num_bits  # noqa: E402
from flcourse.config import load_config, parse_args  # noqa: E402
from flcourse.run import Run  # noqa: E402
from flcourse.secagg import make_secure_aggregator  # noqa: E402


def protocol_label(spec: dict) -> str:
    p = dict(spec)
    name = p.pop("protocol")
    return name + ("(" + ", ".join(f"{k}={v}" for k, v in p.items()) + ")" if p else "")


def costs(bench: Bench, n: int) -> dict:
    """Cost of the busiest client (max over clients: upload, download, computation summed over
    all stages) and the total server computation. Using the maximum (not the mean) keeps
    clients that dropped out -- and therefore sent less -- from hiding the real per-client cost."""
    up, down = defaultdict(int), defaultdict(int)
    for r in bench.comm:
        if kind(r["src"]) == "client":
            up[r["src"]] += r["bits"]
        if kind(r["dst"]) == "client":
            down[r["dst"]] += r["bits"]
    client_time, server_time = defaultdict(float), 0.0
    for r in bench.time:
        if kind(r["who"]) == "client":
            client_time[r["who"]] += r["seconds"]
        else:
            server_time += r["seconds"]
    return {"upload_bits_per_client": float(max(up.values())),
            "download_bits_per_client": float(max(down.values())) if down else 0.0,
            "client_compute_s": max(client_time.values()), "server_compute_s": server_time}


def main(config, overrides=None):
    cfg = load_config(config, overrides)
    sc = cfg.scaling
    run = Run(cfg)
    rng = np.random.default_rng(cfg.seed)
    up_mbps, down_mbps = float(cfg.network.uplink_mbps), float(cfg.network.downlink_mbps)
    rows = []
    for spec in sc.protocols:
        for n in sc.num_clients:
            inputs = {i: rng.uniform(-1, 1, int(sc.dim)) for i in range(int(n))}
            dropped = rng.choice(int(n), size=int(round(float(sc.dropout_rate) * n)), replace=False).tolist()
            stage = "recovery" if spec["protocol"] == "lightsecagg" else "masked_input"
            proto = make_secure_aggregator(spec, seed=int(cfg.seed))
            bench = Bench(f"{spec['protocol']}-{n}").activate()      # one Bench per measurement
            try:
                mean = proto.aggregate(inputs, dropouts={c: stage for c in dropped})
            except (RuntimeError, ValueError) as exc:
                run.log(f"{protocol_label(spec)} n={n}: FAILED ({exc})")
                continue
            finally:
                bench.deactivate()
            survivors = proto.last_info["aggregated_clients"]
            err = float(np.max(np.abs(mean - np.mean([inputs[c] for c in survivors], axis=0))))
            row = {"protocol": protocol_label(spec), "n": int(n), "dim": int(sc.dim), "dropped": len(dropped),
                   "max_error": err, **costs(bench, int(n))}
            row["latency_s"] = bench.latency_estimate(up_mbps, down_mbps)["total_s"]
            rows.append(row)
            run.log(f"{row['protocol']:<55} n={n:>3}  up/client {fmt_bits(row['upload_bits_per_client']):>10}  "
                    f"client {fmt_seconds(row['client_compute_s']):>8}  server {fmt_seconds(row['server_compute_s']):>8}"
                    f"  err {err:.1e}")
    run.bench.activate()
    # plain FedAvg upload for comparison: d float32 values
    plain_bits = num_bits(np.zeros(int(sc.dim), dtype=np.float32))     # size only, not recorded

    # ---- figure -----------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    panels = [("upload_bits_per_client", "upload of the busiest client [MB]", 8e6),
              ("client_compute_s", "computation of the busiest client [s]", 1.0),
              ("server_compute_s", "server computation [s]", 1.0),
              ("latency_s", f"estimated round latency [s]\n({up_mbps:g}/{down_mbps:g} Mbit/s up/down)", 1.0)]
    for ax, (key, title, scale) in zip(axes, panels):
        for proto in dict.fromkeys(r["protocol"] for r in rows):
            pts = [(r["n"], r[key] / scale) for r in rows if r["protocol"] == proto]
            ax.plot(*zip(*pts), "-o", label=proto, ms=4)
        if key == "upload_bits_per_client":
            ax.axhline(plain_bits / scale, color="gray", ls="--", label="plain upload (no SecAgg)")
        ax.set_xlabel("clients per round n")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"secure aggregation of a {sc.dim}-dimensional vector, {float(sc.dropout_rate):.0%} dropouts")
    fig.tight_layout()
    fig.savefig(run.out_dir / "scaling.png", dpi=130)
    plt.close(fig)

    header = ["protocol", "n", "dropped", "upload/client", "download/client", "client compute", "server compute",
              "est. latency", "max error"]
    table = [header] + [[r["protocol"], str(r["n"]), str(r["dropped"]), fmt_bits(r["upload_bits_per_client"]),
                         fmt_bits(r["download_bits_per_client"]), fmt_seconds(r["client_compute_s"]),
                         fmt_seconds(r["server_compute_s"]), fmt_seconds(r["latency_s"]), f"{r['max_error']:.1e}"]
                        for r in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    text = "\n".join("  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(row, widths)))
                     for row in table)
    with open(run.out_dir / "scaling.csv", "w") as fh:
        fh.write(",".join(rows[0].keys()) + "\n")
        for r in rows:
            fh.write(",".join(str(v) for v in r.values()) + "\n")
    return run.finish(notes="SCALING (one aggregation per row)\n" + text + "\n\nfigure: scaling.png",
                      rows=len(rows))


if __name__ == "__main__":
    main(**parse_args(__doc__))
