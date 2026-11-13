"""
Run all demo experiments one after another and write demo_results/SUMMARY.md.

    ** IGNORE THE CONTENT OF THIS FILE. IT IS NOT IMPORTANT FOR UNDERSTANDING THE REPO **

    python experiments/run_demos.py     # everything (~30-40 min on a laptop CPU)
    python experiments/run_demos.py --only fedavg secagg   # names containing these strings
    python experiments/run_demos.py --skip cifar sweep

Each demo is a (script, config, overrides) triple -- exactly what you would type by hand.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib  # noqa: E402

C = "experiments/configs/"
DEMOS = [
    # name,                      script,               config,                          overrides
    ("fedavg_mnist_iid", "fedavg", C + "fedavg_mnist_iid.yaml", []),
    ("fedavg_mnist_dirichlet", "fedavg", C + "fedavg_mnist_dirichlet.yaml", []),
    ("fedsgd_mnist", "fedavg", C + "fedsgd_mnist.yaml", []),
    ("fedavg_cifar10", "fedavg", C + "fedavg_cifar10.yaml", []),
    ("byz_ipm_mean", "byzantine", C + "byz_ipm_mean.yaml", []),
    ("byz_alie_trmean_nnm", "byzantine", C + "byz_alie_mnist.yaml", []),
    ("byz_ipm_fltrust", "byzantine", C + "byz_fltrust_mnist.yaml", []),
    ("byz_labelflip_median", "byzantine", C + "byz_labelflip_mnist.yaml", []),
    ("byz_backdoor_mean", "byzantine", C + "byz_backdoor_mnist.yaml", []),
    ("byz_backdoor_clipping", "byzantine", C + "byz_backdoor_mnist.yaml",
     ["name=byz_backdoor_clipping", "pre_aggregators=[{name: clipping, c: 0.5}]"]),
    ("dp_none_mnist", "dp_fedavg", C + "dp_central_mnist.yaml", ["name=dp_none_mnist", "dp.noise_multiplier=0.0"]),
    ("dp_central_mnist", "dp_fedavg", C + "dp_central_mnist.yaml", []),
    ("dp_local_mnist", "dp_fedavg", C + "dp_local_mnist.yaml", []),
    ("secagg_mnist", "secure_aggregation", C + "secagg_mnist.yaml", []),
    ("secaggplus_mnist", "secure_aggregation", C + "secaggplus_mnist.yaml", []),
    ("lightsecagg_mnist", "secure_aggregation", C + "lightsecagg_mnist.yaml", []),
    ("lightsecagg_mnist_dropouts", "secure_aggregation", C + "lightsecagg_mnist_dropouts.yaml", []),
    ("secagg_dp_mnist", "secure_aggregation", C + "secagg_dp_mnist.yaml", []),
    ("secagg_scaling", "secagg_scaling", C + "secagg_scaling.yaml", []),
    ("privacy_attacks_mnist", "privacy_attacks", C + "privacy_attacks_mnist.yaml", []),
    ("privacy_attacks_mnist_central", "privacy_attacks", C + "privacy_attacks_mnist.yaml",
     ["name=privacy_attacks_mnist_central", "privacy.dp_mode=central"]),
    ("sweep_byzantine_mnist", "sweep", C + "sweep_byzantine_mnist.yaml", []),
    ("sweep_byzantine_noniid_mnist", "sweep", C + "sweep_byzantine_noniid_mnist.yaml", []),
    ("sweep_byzantine_noniid_nnm_mnist", "sweep", C + "sweep_byzantine_noniid_nnm_mnist.yaml", []),
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--skip", nargs="*", default=[])
    parser.add_argument("--output", default="demo_results")
    parser.add_argument("--set", nargs="*", default=[], help="extra overrides for every demo, e.g. device=cpu")
    args = parser.parse_args(argv)
    out = Path(args.output)
    out.mkdir(exist_ok=True)
    log = {}
    if (out / "runs.json").exists():
        log = json.loads((out / "runs.json").read_text())
    for name, script, config, overrides in DEMOS:
        if args.only and not any(s in name for s in args.only):
            continue
        if any(s in name for s in args.skip):
            continue
        print(f"\n######## {name}: python experiments/{script}.py --config {config} "
              f"{'--set ' + ' '.join(overrides) if overrides else ''}", flush=True)
        t0 = time.time()

        # we actually implant the "main" method of the various python files in here 
        module = importlib.import_module(f"experiments.{script}")
        try:
            summary = module.main(config, overrides=[f"output_dir={out}", *overrides, *args.set])
            log[name] = {"script": script, "config": config, "overrides": overrides, "ok": True,
                         "seconds": time.time() - t0, "output_dir": summary.get("output_dir"),
                         "final": summary.get("final"), "summary": {k: v for k, v in summary.items()
                                                                     if k not in ("final", "results")}}
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            log[name] = {"script": script, "config": config, "ok": False, "error": repr(exc),
                         "seconds": time.time() - t0}
        (out / "runs.json").write_text(json.dumps(log, indent=1, default=str))
    from experiments import summarize_demos
    summarize_demos.main(["--output", str(out)])
    print(f"\nDone. Index of all runs: {out / 'runs.json'}, overview: {out / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
