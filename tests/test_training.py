"""The three upload modes must be consistent with each other."""
import numpy as np
import torch

from flcourse.aggregators import Average
from flcourse.client import Client
from flcourse.config import Config
from flcourse.data import load_dataset, partition
from flcourse.models import build_model, get_flat_params, set_flat_params
from flcourse.server import apply_aggregate, num_selected, sample_clients
from flcourse.training import evaluate


def _setup(local_cfg):
    train, test = load_dataset({"dataset": "synthetic"})
    parts = partition(train.y.numpy(), 4, "iid", seed=0)
    clients = [Client(i, train.subset(p), Config(local_cfg), seed=0) for i, p in enumerate(parts)]
    torch.manual_seed(0)
    model = build_model("mlp", train.input_shape, 10)
    return clients, model, get_flat_params(model), test


def test_fedsgd_equals_one_full_batch_step_of_fedavg():
    """gradient mode with server_lr = lr  ==  delta/model mode with ONE full-batch local SGD step."""
    lr = 0.1
    clients, model, w0, _ = _setup({"batch_size": 10_000, "gradient_batches": 1, "local_steps": 1, "lr": lr})
    grads = [c.local_update(model, w0, "gradient")[0] for c in clients]
    deltas = [c.local_update(model, w0, "delta")[0] for c in clients]
    models = [c.local_update(model, w0, "model")[0] for c in clients]
    w_grad = apply_aggregate(w0, Average()(grads), "gradient", server_lr=lr)
    w_delta = apply_aggregate(w0, Average()(deltas), "delta", server_lr=1.0)
    w_model = apply_aggregate(w0, Average()(models), "model")
    torch.testing.assert_close(w_grad, w_delta, atol=1e-6, rtol=0)
    torch.testing.assert_close(w_delta, w_model, atol=1e-6, rtol=0)


def test_numpy_backend_gives_the_same_update():
    clients, model, w0, _ = _setup({"batch_size": 10_000, "local_steps": 1, "lr": 0.1})
    deltas = [c.local_update(model, w0, "delta")[0] for c in clients]
    a = apply_aggregate(w0, Average()(deltas), "delta")
    b = apply_aggregate(w0, Average()([d.numpy() for d in deltas]), "delta")
    torch.testing.assert_close(a, b)


def test_local_training_reduces_the_loss():
    clients, model, w0, test = _setup({"batch_size": 32, "local_epochs": 2, "lr": 0.1})
    set_flat_params(model, w0)
    loss0, _ = evaluate(model, test)
    w = w0
    for _ in range(3):
        w = apply_aggregate(w, Average()([c.local_update(model, w, "model")[0] for c in clients]), "model")
    set_flat_params(model, w)
    loss1, acc1 = evaluate(model, test)
    assert loss1 < 0.5 * loss0 and acc1 > 0.8


def test_global_vector_is_not_modified_by_local_training():
    clients, model, w0, _ = _setup({"batch_size": 32, "local_epochs": 1, "lr": 0.1})
    before = w0.clone()
    clients[0].local_update(model, w0, "model")
    assert torch.equal(w0, before)


def test_client_sampling():
    rng = np.random.default_rng(0)
    sel = sample_clients(20, 5, rng)
    assert len(sel) == 5 and len(set(sel)) == 5 and sel == sorted(sel)
    assert num_selected(20, 0.25) == 5 and num_selected(20, 1.0) == 20 and num_selected(20, 3) == 3
