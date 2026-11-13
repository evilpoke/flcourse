"""Run an experiment for every combination of a parameter grid and draw a heatmap.

    python experiments/sweep.py --config experiments/configs/sweep_byzantine_mnist.yaml

Sweep config::

    name: sweep_byzantine_mnist
    experiment: byzantine                 # a script in experiments/ (without .py)
    base_config: byz_base_mnist.yaml      # path relative to this sweep file
    metric: test_acc                      # read from the last evaluated round
    grid:                                 # 1 or 2 dotted keys; values REPLACE the base entry
      attack.vector: [null, {name: sign_flipping}, {name: alie, tau: 1.5}]
      aggregator: [{name: mean}, {name: median}, {name: krum}]
    overrides: {fl: {rounds: 10}}         # optional, applied to every run

Every run writes its own result folder (``<sweep output>/runs/...``); the sweep folder
contains sweep.csv, heatmap.png and report.txt.
"""
import copy
import importlib
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from flcourse.config import deep_merge, parse_args  # noqa: E402


ABBREVIATIONS = {"centered_clipping": "cclip", "geometric_median": "geomed", "trimmed_mean": "trmean",
                 "sign_flipping": "signflip", "inner_product_manipulation": "ipm", "a_little_is_enough": "alie"}


def short(value) -> str:
    """Readable name of a grid value: {name: alie, tau: 1.5} -> 'alie(tau=1.5)'."""
    if value is None:
        return "none"
    if isinstance(value, dict):
        rest = {k: v for k, v in value.items() if k != "name"}
        params = ",".join(f"{k}={v}" for k, v in rest.items())
        name = str(value.get("name", "?"))
        return ABBREVIATIONS.get(name, name) + (f"({params})" if params else "")
    return str(value)


def set_path(cfg: dict, path: str, value) -> None:
    node = cfg
    parts = path.split(".")
    for p in parts[:-1]:
        if node.get(p) is None:
            node[p] = {}
        node = node[p]
    node[parts[-1]] = copy.deepcopy(value)


def main(config, overrides=None):
    sweep_path = Path(config)
    sweep = yaml.safe_load(sweep_path.read_text())
    if overrides:
        from flcourse.config import _parse_override
        for item in overrides:
            sweep = deep_merge(sweep, _parse_override(item))
    module = importlib.import_module(f"experiments.{sweep['experiment']}")
    base = yaml.safe_load((sweep_path.parent / sweep["base_config"]).read_text())
    base = deep_merge(base, sweep.get("overrides") or {})
    keys = list(sweep["grid"].keys())
    if not 1 <= len(keys) <= 2:
        raise ValueError("grid must have one or two keys")
    root = sweep.get("output_dir") or base.get("output_dir", "results")
    out = Path(root) / sweep.get("name", "sweep") / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    metric = sweep.get("metric", "test_acc")

    results = {}
    for combo in itertools.product(*(sweep["grid"][k] for k in keys)):
        cfg = copy.deepcopy(base)
        for k, v in zip(keys, combo):
            set_path(cfg, k, v)
        tag = " | ".join(f"{k}={short(v)}" for k, v in zip(keys, combo))
        cfg["name"] = "__".join(short(v).replace("/", "_") for v in combo)
        cfg["output_dir"] = str(out / "runs")
        cfg["verbose"] = bool(sweep.get("verbose", False))
        t0 = time.time()
        try:
            summary = module.main(cfg)
            value = summary["final"].get(metric, float("nan"))
        except Exception as exc:  # noqa: BLE001 - e.g. a rule that needs more clients
            print(f"[sweep] {tag}: FAILED ({exc})", flush=True)
            value = float("nan")
        results[tuple(short(v) for v in combo)] = value
        print(f"[sweep] {tag:<80} {metric} = {value:.4f}   ({time.time() - t0:.0f}s)", flush=True)

    rows = [short(v) for v in sweep["grid"][keys[0]]]
    cols = [short(v) for v in sweep["grid"][keys[1]]] if len(keys) == 2 else [metric]
    M = np.array([[results[(r, c)] if len(keys) == 2 else results[(r,)] for c in cols] for r in rows])

    with open(out / "sweep.csv", "w") as fh:
        fh.write(f"{keys[0]}," + ",".join(cols) + "\n")
        for r, row in zip(rows, M):
            fh.write(r + "," + ",".join(f"{v:.4f}" for v in row) + "\n")

    fig, ax = plt.subplots(figsize=(1.6 * len(cols) + 2.5, 0.55 * len(rows) + 1.8))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1 if "acc" in metric else None, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    ax.set_xlabel(keys[1] if len(keys) == 2 else "")
    ax.set_ylabel(keys[0])
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, "fail" if np.isnan(M[i, j]) else f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label=metric)
    ax.set_title(f"{sweep.get('name', 'sweep')}: final {metric}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "heatmap.png", dpi=140)
    plt.close(fig)

    width = max(len(r) for r in rows) + 2
    lines = [f"SWEEP {sweep.get('name')}  ({sweep['experiment']}.py, base {sweep['base_config']}), final {metric}",
             " " * width + "".join(c[:18].rjust(20) for c in cols)]
    lines += [r.ljust(width) + "".join(("fail" if np.isnan(v) else f"{v:.4f}").rjust(20) for v in row)
              for r, row in zip(rows, M)]
    text = "\n".join(lines)
    (out / "report.txt").write_text(text + "\n")
    print("\n" + text + f"\n\nresults: {out}")
    return {"output_dir": str(out), "results": {"/".join(k): v for k, v in results.items()}}


if __name__ == "__main__":
    main(**parse_args(__doc__))
