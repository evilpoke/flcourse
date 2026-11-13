import numpy as np
import pytest
import torch

from flcourse.attacks import (ATTACKS, ALittleIsEnough, CollusionChannel, Gaussian, Inf, InnerProductManipulation,
                              MinMax, MinSum, Mimic, Scaling, SignFlipping, Zero, add_trigger, backdoor_testset,
                              flip_labels, infer_label_idlg, invert_gradient, label_flipping, make_attack,
                              poison_backdoor)
from flcourse.backend import pairwise_sq_distances
from flcourse.bench import Bench
from flcourse.data.datasets import ArrayDataset
from flcourse.models import build_model
from flcourse.training import compute_gradient

DETERMINISTIC = [SignFlipping(), InnerProductManipulation(2.0), ALittleIsEnough(1.5), Mimic(1), Zero(), Scaling(3.0),
                 MinMax("std"), MinMax("unit"), MinMax("sign"), MinSum("std")]


@pytest.mark.parametrize("attack", DETERMINISTIC, ids=repr)
def test_numpy_torch_parity(attack, rng):
    X = rng.standard_normal((8, 12))
    a, b = attack(X), attack(torch.tensor(X))
    assert a.shape == (12,) and isinstance(b, torch.Tensor)
    np.testing.assert_allclose(a, b.numpy(), atol=1e-10)


def test_formulas(rng):
    X = rng.standard_normal((8, 12))
    np.testing.assert_allclose(SignFlipping()(X), -X.mean(0))
    np.testing.assert_allclose(InnerProductManipulation(3.0)(X), -3.0 * X.mean(0))
    np.testing.assert_allclose(ALittleIsEnough(1.5)(X), X.mean(0) + 1.5 * X.std(0, ddof=1))
    np.testing.assert_allclose(Scaling(4.0)(X), 4.0 * X.mean(0))
    np.testing.assert_array_equal(Mimic(2)(X), X[2])
    assert np.all(Zero()(X) == 0) and np.all(np.isinf(Inf()(X)))
    assert Gaussian(sigma=2.0, seed=0)(X).shape == (12,)
    assert Gaussian(seed=0)(torch.tensor(X)).shape == (12,)


def test_minmax_respects_its_constraint(rng):
    X = rng.standard_normal((10, 30))
    mal = MinMax("std")(X)
    max_honest = pairwise_sq_distances(X).max()
    assert ((X - mal) ** 2).sum(1).max() <= max_honest + 1e-9
    assert np.linalg.norm(mal - X.mean(0)) > 0.1          # it does move away from the mean
    mal_sum = MinSum("std")(X)
    assert ((X - mal_sum) ** 2).sum() <= pairwise_sq_distances(X).sum(1).max() + 1e-9


def test_single_attacker_falls_back_to_honest():
    x = np.ones((1, 4))
    np.testing.assert_array_equal(ALittleIsEnough()(x), x[0])
    np.testing.assert_array_equal(MinMax()(x), x[0])


def test_make_attack():
    assert make_attack(None) is None and make_attack("none") is None
    assert isinstance(make_attack({"name": "alie", "tau": 2.0}), ALittleIsEnough)
    with pytest.raises(ValueError):
        make_attack("nope")
    assert {"sign_flipping", "ipm", "alie", "minmax", "gaussian", "scaling"} <= set(ATTACKS)


def _images(n=20):
    x = torch.randint(0, 100, (n, 1, 16, 16), dtype=torch.uint8)
    y = torch.arange(n) % 10
    return ArrayDataset(x, y, 10, mean=(0.1,), std=(0.3,))


def test_label_flipping_and_backdoor():
    ds = _images()
    np.testing.assert_array_equal(flip_labels(ds.y, 10).numpy(), 9 - ds.y.numpy())
    assert torch.equal(label_flipping(ds).y, 9 - ds.y)
    assert torch.equal(label_flipping(ds, source=3, target=8).y == 8, (ds.y == 3) | (ds.y == 8))
    trig = add_trigger(ds.x, size=3)
    assert torch.all(trig[:, :, -4:-1, -4:-1] == 255) and torch.equal(trig[:, :, :8], ds.x[:, :, :8])
    poisoned = poison_backdoor(ds, target_label=0, fraction=0.5, size=3)
    changed = (poisoned.x != ds.x).flatten(1).any(1)
    assert changed.sum() >= 1 and torch.all(poisoned.y[changed] == 0)
    bd = backdoor_testset(ds, target_label=0)
    assert len(bd) == int((ds.y != 0).sum()) and torch.all(bd.y == 0)
    feats = ArrayDataset(torch.zeros(4, 6), torch.zeros(4), 2)
    assert torch.all(add_trigger(feats.x, size=2)[:, -2:] == 5.0)


def test_collusion_channel_records_communication():
    bench = Bench().activate()
    try:
        ch = CollusionChannel()
        ch.post(1, np.zeros(10, dtype=np.float32))
        ch.post(2, np.ones(10, dtype=np.float32))
        ch.post(3, b"secret", topic="shares", record=False)
        assert ch.senders() == [1, 2] and len(ch.values()) == 2 and ch.read("shares") == {3: b"secret"}
        assert len(ch) == 3
        assert [r["bits"] for r in bench.comm] == [320, 320]
        assert bench.comm[0]["label"] == "collusion/vector" and bench.comm[0]["dst"] == "colluders"
    finally:
        bench.deactivate()


def test_gradient_inversion_recovers_a_single_image():
    torch.manual_seed(0)
    model = build_model("dlg_lenet", (1, 16, 16), 10)
    data = _images(1)
    grad, _ = compute_gradient(model, data, batch_size=1, num_batches=1)
    label = infer_label_idlg(model, grad)
    assert label == int(data.y[0])
    x_rec, labels, history = invert_gradient(model, grad, (1, 16, 16), 10, label=label, iterations=15)
    x_true = data.batch(torch.arange(1))[0]
    assert history[-1] < 1e-3 * history[0]
    assert float(((x_rec - x_true) ** 2).mean()) < 0.05 * float((x_true ** 2).mean())
