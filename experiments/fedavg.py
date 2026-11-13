"""Baseline federated learning: FedAvg (upload: model / delta) and FedSGD (upload: gradient).

Example:

    python experiments/fedavg.py --config experiments/configs/fedavg_mnist_iid.yaml

The other experiments are variations of this.


Communication is recorded with ``measure(...)`` and computation with ``timer(...)``
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # run without `pip install -e .`

import numpy as np  # noqa: E402

from flcourse.aggregators import make_aggregator  # noqa: E402
from flcourse.backend import stack, to_backend  # noqa: E402
from flcourse.bench import measure, timer  # noqa: E402
from flcourse.config import load_config, parse_args  # noqa: E402
from flcourse.dp import add_gaussian_noise, clip_l2  # noqa: E402,F401  (used in commented lines)
from flcourse.models import set_flat_params  # noqa: E402
from flcourse.preaggregators import make_pre_aggregators  # noqa: E402
from flcourse.run import Run, build_federation  # noqa: E402
from flcourse.server import apply_aggregate, sample_clients  # noqa: E402
from flcourse.training import evaluate  # noqa: E402


def main(config, overrides=None):
    cfg = load_config(config, overrides)
    run = Run(cfg)

    fed = build_federation(cfg, run) # Generates a "federation": This holds the data (and partitioning of it) and the clients

    pre_aggregators = make_pre_aggregators(cfg.pre_aggregators)   # Note that with pre aggregation like NNM, we can make a robust aggregation scheme more effective (see e.g. lectures about prodigy)

    aggregator = make_aggregator(cfg.aggregator)  # Selecting the actual robust aggregation scheme. We use FedAvg here (if cfg is still with default values)

    rng = np.random.default_rng(cfg.seed)   # This randomness handles client selection only and NOT training. Note that certain DL optimizers use randomness in the gradient estimation as part of their regularisation. Consider this whenever you want training curves to be perfectly reproducible

    upload_mode = cfg.fl.upload                       # "model" | "delta" | "gradient"

    global_vec = fed.global_vector                    # the initial global model to be broadcasted.

    run.log(f"aggregator: {aggregator}   pre-aggregators: {pre_aggregators}   upload: {upload_mode}")

    for rnd in run.rounds():
        # note that not all clients may be considered.
        selected = sample_clients(cfg.fl.num_clients, cfg.fl.clients_per_round, rng)

        # ---- each client contributes -----------------------------
        uploads = []
        for cid in selected:
            client = fed.clients[cid]

            # we track the upload size of the FL material
            measure(global_vec, "download: global model", src="server", dst=cid)
            # we also track the time
            with timer("client: local update", who=cid):
                upload, stats = client.local_update(fed.model, global_vec, upload_mode)
            # we wrap the upload material into whatever we are working with (torch/np)
            upload = to_backend(upload, cfg.backend)     

            # ---- 2. client-side post-processing of the upload material. This can be applied to all of delta, gradient, model. Just comment it out       
            # upload = clip_l2(upload, max_norm=1.0)                      # << norm bounding
            # upload = add_gaussian_noise(upload, std=0.01)               # << (local) noise

            measure(upload, f"upload: {upload_mode}", src=cid, dst="server")
            uploads.append(upload)
            run.log_client(rnd, cid, **stats)

        # ---- server aggregates -----------------------
        with timer("server: aggregate"):
            X = stack(uploads)                #  <<(n_selected, d) matrix
            for pre in pre_aggregators:       # <<e.g. NNM, Bucketing, Clipping
                X = pre(X)
            weights = [fed.clients[c].num_samples for c in selected]
            aggregate = aggregator(X, weights=weights)            # e.g. mean, median, krum
            global_vec = apply_aggregate(global_vec, aggregate, upload_mode, cfg.fl.server_lr)

        # ---- evaluation --------------------------
        if rnd % cfg.eval.every == 0 or rnd == cfg.fl.rounds:
            with timer("server: evaluate"):
                set_flat_params(fed.model, global_vec)
                loss, acc = evaluate(fed.model, fed.test, cfg.eval.batch_size, run.device)
            run.log_global(rnd, test_loss=loss, test_acc=acc)
    
    return run.finish()


if __name__ == "__main__":
    main(**parse_args(__doc__))
