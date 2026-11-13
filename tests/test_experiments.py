"""End-to-end runs of every experiment script on tiny offline settings (CPU, no download)."""
import json

import pytest

from experiments import (byzantine, dp_fedavg, fedavg, privacy_attacks, secagg_scaling, secure_aggregation,
                         sweep)

CONFIGS = "experiments/configs/"


def tiny(tmp_path, **extra):
    base = {"output_dir": str(tmp_path), "verbose": False, "device": "cpu",
            "data": {"dataset": "digits"}, "model": {"name": "mlp"},
            "fl": {"num_clients": 8, "clients_per_round": 8, "rounds": 3},
            "client": {"local_steps": 5, "batch_size": 16, "lr": 0.1}}
    for key, value in extra.items():
        base[key] = {**base.get(key, {}), **value} if isinstance(value, dict) else value
    return base


def _outputs_exist(summary):
    from pathlib import Path
    out = Path(summary["output_dir"])
    for name in ("report.txt", "metrics.json", "config.yaml", "convergence.png", "communication.png"):
        assert (out / name).exists(), name
    return json.loads((out / "metrics.json").read_text())


@pytest.mark.parametrize("upload", ["model", "delta", "gradient"])
def test_fedavg_learns(tmp_path, upload):
    cfg = tiny(tmp_path, fl={"upload": upload, "server_lr": 0.5 if upload == "gradient" else 1.0, "rounds": 5},
               client={"gradient_batches": -1})
    summary = fedavg.main(f"{CONFIGS}quick_digits.yaml", overrides=cfg)
    metrics = _outputs_exist(summary)
    assert summary["final"]["test_acc"] > 0.5                 # 10 classes -> chance is 0.1
    assert len(metrics["clients"]) == 5 * 8


def test_fedavg_numpy_backend_and_preaggregators(tmp_path):
    cfg = tiny(tmp_path, backend="numpy", aggregator={"name": "trimmed_mean", "f": 1},
               pre_aggregators=[{"name": "nnm", "f": 1}])
    summary = fedavg.main(f"{CONFIGS}quick_digits.yaml", overrides=cfg)
    assert summary["final"]["test_acc"] > 0.3


@pytest.mark.parametrize("config", ["byz_alie_mnist", "byz_fltrust_mnist", "byz_backdoor_mnist",
                                    "byz_labelflip_mnist", "byz_ipm_mean"])
def test_byzantine_configs_run(tmp_path, config):
    cfg = tiny(tmp_path, attack={"num_byzantine": 2})
    summary = byzantine.main(f"{CONFIGS}{config}.yaml", overrides=cfg)
    assert 0.0 <= summary["final"]["test_acc"] <= 1.0
    assert summary["byzantine"] and len(summary["byzantine"]) == 2


def test_ipm_breaks_the_mean_but_not_the_median(tmp_path):
    cfg = tiny(tmp_path, fl={"rounds": 8}, client={"local_steps": 10},
               attack={"num_byzantine": 2, "vector": {"name": "ipm", "tau": 5.0}})
    mean = byzantine.main(f"{CONFIGS}byz_ipm_mean.yaml", overrides={**cfg, "aggregator": {"name": "mean"}})
    med = byzantine.main(f"{CONFIGS}byz_ipm_mean.yaml", overrides={**cfg, "aggregator": {"name": "median"}})
    assert med["final"]["test_acc"] > mean["final"]["test_acc"] + 0.2


@pytest.mark.parametrize("mode", ["central", "local", "distributed"])
def test_dp(tmp_path, mode):
    cfg = tiny(tmp_path, dp={"mode": mode, "clip": 1.0, "noise_multiplier": 0.5, "delta": 1e-5})
    summary = dp_fedavg.main(f"{CONFIGS}dp_central_mnist.yaml", overrides=cfg)
    assert summary["epsilon"] > 0 and summary["final"]["epsilon"] == summary["epsilon"]


def test_dp_refuses_model_uploads(tmp_path):
    with pytest.raises(ValueError):
        dp_fedavg.main(f"{CONFIGS}dp_central_mnist.yaml", overrides=tiny(tmp_path, fl={"upload": "model"}))


@pytest.mark.parametrize("config", ["secagg_mnist", "secaggplus_mnist", "lightsecagg_mnist", "secagg_dp_mnist"])
def test_secure_aggregation(tmp_path, config):
    cfg = tiny(tmp_path, fl={"rounds": 2})
    summary = secure_aggregation.main(f"{CONFIGS}{config}.yaml", overrides=cfg)
    assert summary["final"]["secagg_max_error"] < 1e-4
    _outputs_exist(summary)


def test_lightsecagg_two_dropout_filters(tmp_path):
    """filter 1: dropped before uploading (not in U_1); filter 2: in U_1 but no recovery message."""
    summary = secure_aggregation.main(f"{CONFIGS}lightsecagg_mnist_dropouts.yaml", overrides=tiny(tmp_path, fl={"rounds": 2}))
    final = summary["final"]                              # N = 8 per round, T = 2, U = 6
    assert final["survivors"] == 7 and final["recovery_responses"] == 6
    assert final["secagg_max_error"] < 1e-4
    with pytest.raises(ValueError, match="only defined for protocol: lightsecagg"):
        secure_aggregation.main(f"{CONFIGS}secagg_mnist.yaml", overrides=tiny(
            tmp_path, secagg={"recovery_dropout_rate": 0.2}))


def test_secure_aggregation_refuses_robust_rules(tmp_path):
    with pytest.raises(ValueError, match="only the SUM"):
        secure_aggregation.main(f"{CONFIGS}secagg_mnist.yaml", overrides=tiny(tmp_path, aggregator={"name": "krum"}))


def test_secagg_scaling(tmp_path):
    summary = secagg_scaling.main(f"{CONFIGS}secagg_scaling.yaml", overrides={
        "output_dir": str(tmp_path), "verbose": False, "scaling": {"num_clients": [4, 6], "dim": 200}})
    assert summary["rows"] == 6


@pytest.mark.parametrize("mode", ["none", "local", "central"])
def test_privacy_attacks(tmp_path, mode):
    cfg = {"output_dir": str(tmp_path), "verbose": False, "data": {"dataset": "digits"},
           "fl": {"num_clients": 6, "clients_per_round": 4},
           "privacy": {"iterations": 2, "dp_noise_multipliers": [0.01, 0.1], "dp_mode": mode}}
    summary = privacy_attacks.main(f"{CONFIGS}privacy_attacks_mnist.yaml", overrides=cfg)
    from pathlib import Path
    assert (Path(summary["output_dir"]) / "gradient_inversion.png").exists()
    assert (Path(summary["output_dir"]) / "aggregate_all_clients.png").exists()
    assert "no DP" in summary["psnr_individual"] and "no DP" in summary["psnr_aggregate"]
    # the joint inversion of the aggregate yields one dummy per participant, in every setting
    assert all(len(row) == 4 for row in summary["psnr_aggregate"].values())
    assert summary["psnr_aggregate"].keys() == summary["psnr_individual"].keys()
    if mode == "none":
        assert len(summary["psnr_individual"]) == 1          # no sigma columns
    else:
        assert "sigma=0.01" in summary["psnr_individual"] and "sigma=0.1" in summary["psnr_aggregate"]
    if mode == "central":                                    # server noise comes after the upload:
        assert summary["psnr_individual"]["sigma=0.01"] == summary["psnr_individual"]["sigma=0.1"]


def test_sweep(tmp_path):
    sweep_cfg = tmp_path / "sweep.yaml"
    base = tmp_path / "base.yaml"
    import yaml
    base.write_text(yaml.safe_dump(tiny(tmp_path, fl={"rounds": 2})))
    sweep_cfg.write_text(yaml.safe_dump({
        "name": "s", "experiment": "fedavg", "base_config": "base.yaml", "output_dir": str(tmp_path),
        "grid": {"aggregator": [{"name": "mean"}, {"name": "median"}], "fl.upload": ["model", "delta"]}}))
    out = sweep.main(str(sweep_cfg))
    assert len(out["results"]) == 4
