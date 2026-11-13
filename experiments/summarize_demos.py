"""

    ** IGNORE THE CONTENT OF THIS FILE. IT IS NOT IMPORTANT FOR UNDERSTANDING THE REPO **


Write demo_results/SUMMARY.md from the runs recorded by run_demos.py.

    python experiments/summarize_demos.py [--output demo_results] [--replot]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flcourse.bench import fmt_bits, fmt_seconds  # noqa: E402
from flcourse.report import replot  # noqa: E402

FIGURES = ["convergence.png", "partition.png", "communication.png", "computation.png", "scaling.png",
           "gradient_inversion.png", "heatmap.png"]


def fmt(v, digits=4):
    if isinstance(v, float):
        return f"{v:.1e}" if 0 < abs(v) < 1e-3 else f"{v:.{digits}f}"
    return "" if v is None else str(v)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="demo_results")
    parser.add_argument("--replot", action="store_true", help="re-draw the figures of every run first")
    args = parser.parse_args(argv)
    out = Path(args.output)
    runs = json.loads((out / "runs.json").read_text())
    lines = ["# Demo results", "",
             "Produced by `python experiments/run_demos.py` on a 4-core CPU without GPU "
             "(the numbers are from one seed; timings are of the sequential simulation).", "",
             "| demo | command | test acc | other | rounds | communication | wall clock |",
             "|---|---|---|---|---|---|---|"]
    for name, r in runs.items():
        cmd = f"`python experiments/{r['script']}.py --config {r['config']}" + \
              (f" --set {' '.join(r.get('overrides') or [])}" if r.get("overrides") else "") + "`"
        if not r.get("ok"):
            lines.append(f"| {name} | {cmd} | FAILED: {r.get('error')} | | | | |")
            continue
        final = r.get("final") or {}
        s = r.get("summary") or {}
        other = []
        for key in ("backdoor_asr", "epsilon", "secagg_max_error"):
            if key in final:
                other.append(f"{key}={fmt(final[key], 3)}")
        comm = s.get("total_communication_bits") or 0
        lines.append(f"| {name} | {cmd} | {fmt(final.get('test_acc'))} | {', '.join(other)} | "
                     f"{final.get('round', '')} | {fmt_bits(comm) if comm else ''} | "
                     f"{fmt_seconds(r.get('seconds', 0))} |")
    lines.append("")
    for name, r in runs.items():
        if not r.get("ok") or not r.get("output_dir"):
            continue
        d = Path(r["output_dir"])
        if args.replot and (d / "metrics.json").exists():
            replot(d)
        rel = d.relative_to(out) if d.is_relative_to(out) else d
        lines += [f"## {name}", "", f"folder: `{rel}`", ""]
        report = d / "report.txt"
        if report.exists():
            lines += ["<details><summary>report.txt</summary>", "", "```", report.read_text().rstrip(), "```",
                      "", "</details>", ""]
        for fig in FIGURES:
            if (d / fig).exists():
                lines.append(f"![{name} {fig}]({rel}/{fig})")
        lines.append("")
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
