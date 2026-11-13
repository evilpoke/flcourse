import warnings

import numpy as np
import pytest
import torch

from flcourse.config import load_config
from flcourse.data import label_distribution, load_dataset, partition
from flcourse.device import get_device
from flcourse.models import build_model, get_flat_params, num_params, set_flat_params


@pytest.fixture
def labels(rng):
    return rng.integers(0, 10, 5000)


def _check_disjoint_cover(parts, n):
    allidx = np.concatenate(parts)
    assert len(allidx) == n and len(np.unique(allidx)) == n


def test_iid_partition(labels):
    parts = partition(labels, 10, "iid", seed=0)
    _check_disjoint_cover(parts, len(labels))
    assert {len(p) for p in parts} == {500}


def test_dirichlet_partition_is_skewed(labels):
    parts = partition(labels, 10, "dirichlet", alpha=0.1, min_size=20, seed=0)
    _check_disjoint_cover(parts, len(labels))
    assert min(len(p) for p in parts) >= 20
    counts = label_distribution(labels, parts, 10)
    dominant = (counts.max(1) / counts.sum(1)).mean()
    iid_counts = label_distribution(labels, partition(labels, 10, "iid"), 10)
    assert dominant > 2 * (iid_counts.max(1) / iid_counts.sum(1)).mean()


def test_shard_partition(labels):
    parts = partition(labels, 10, "shards", shards_per_client=2, seed=0)
    _check_disjoint_cover(parts, len(labels))
    classes_per_client = (label_distribution(labels, parts, 10) > 0).sum(1)
    assert classes_per_client.max() <= 4           # 2 shards, each spans at most 2 classes


def test_offline_datasets():
    train, test = load_dataset({"dataset": "synthetic"})
    assert train.input_shape == (20,) and len(test) == 1000
    xb, yb = train.batch(torch.arange(5))
    assert xb.dtype == torch.float32 and yb.dtype == torch.long
    dtrain, dtest = load_dataset({"dataset": "digits", "max_train_samples": 500})
    assert dtrain.input_shape == (1, 8, 8) and len(dtrain) == 500 and len(dtest) == 360
    xb, _ = dtrain.batch(torch.arange(len(dtrain)))
    assert abs(float(xb.mean())) < 0.05 and abs(float(xb.std()) - 1) < 0.05   # normalised
    sub = dtrain.subset([0, 2, 4])
    assert len(sub) == 3 and torch.equal(sub.y, dtrain.y[[0, 2, 4]])


@pytest.mark.parametrize("name,shape", [("logreg", (1, 28, 28)), ("mlp", (1, 8, 8)), ("lenet5", (1, 28, 28)),
                                        ("lenet5", (3, 32, 32)), ("cnn", (3, 32, 32)), ("cnn_large", (1, 28, 28)),
                                        ("dlg_lenet", (1, 28, 28)), ("resnet18", (3, 32, 32))])
def test_models_forward(name, shape):
    model = build_model({"name": name}, shape, 10)
    assert model(torch.zeros(2, *shape)).shape == (2, 10)
    assert num_params(model) == get_flat_params(model).numel()


def test_lenet_rejects_tiny_images():
    with pytest.raises(ValueError):
        build_model("lenet5", (1, 8, 8), 10)


def test_flat_params_roundtrip_without_aliasing():
    model = build_model("mlp", (1, 8, 8), 10)
    vec = torch.randn(num_params(model))
    set_flat_params(model, vec)
    assert torch.equal(get_flat_params(model), vec)
    with torch.no_grad():
        next(model.parameters()).add_(1.0)                     # training must not change `vec`
    assert not torch.equal(get_flat_params(model), vec)
    set_flat_params(model, vec.numpy())                        # numpy vectors work too
    assert torch.equal(get_flat_params(model), vec)
    with pytest.raises(ValueError):
        set_flat_params(model, torch.zeros(num_params(model) + 1))


def test_device_selection_falls_back_to_cpu():
    assert get_device("cpu").type == "cpu"
    dev = get_device("auto")
    assert torch.ones(2, device=dev).sum().item() == 2
    if not torch.cuda.is_available():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert get_device("cuda").type == "cpu"
        assert any("falling back" in str(w.message) for w in caught)


def test_config_defaults_and_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("name: t\nfl:\n  rounds: 7\n")
    cfg = load_config(p, overrides=["fl.upload=gradient", "aggregator={name: krum, f: 2}", "seed=3"])
    assert cfg.fl.rounds == 7 and cfg.fl.upload == "gradient" and cfg.seed == 3
    assert cfg.aggregator.f == 2 and cfg.fl.num_clients == 10                      # default kept
    assert cfg.get_path("data.partition") == "iid" and cfg.get_path("nope.x", 5) == 5
    with pytest.raises(AttributeError):
        cfg.does_not_exist
