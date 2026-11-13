"""Byzantine-robust FL: attacks by colluding clients vs. robust aggregation rules.

    python experiments/byzantine.py --config experiments/configs/byz_alie_vs_median.yaml

Compared with fedavg.py there are three additions:

* data poisoning (label flipping / backdoor) of the Byzantine clients' local data,
* a second loop over the Byzantine clients of the round that talk over a shared
  ``CollusionChannel`` and jointly craft the malicious upload (ByzFL interface:
  ``attack(honest_vectors) -> byzantine_vector``),
* a virtual server-side client; in case the server requires a reference dataset for robustness  (e.g. FLTrust)


Config section ``attack``:

    attack:
      num_byzantine: 4          # how many of the clients are Byzantine (fixed set)
      f: 4                      # what the robust rules assume (default: num_byzantine)
      data: null                # null | label_flip | backdoor
      vector: {name: alie, tau: 1.5}   # null or any attack in flcourse.attacks.ATTACKS
      omniscient: false         # true: attackers also see all honest uploads of the round
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import warnings  # noqa: E402

from flcourse.aggregators import make_aggregator  # noqa: E402
from flcourse.attacks import CollusionChannel, backdoor_testset, label_flipping, make_attack, poison_backdoor  # noqa
from flcourse.backend import as_matrix, stack, to_backend  # noqa: E402
from flcourse.bench import measure, timer  # noqa: E402
from flcourse.client import Client  # noqa: E402
from flcourse.config import load_config, parse_args  # noqa: E402
from flcourse.models import set_flat_params  # noqa: E402
from flcourse.preaggregators import make_pre_aggregators  # noqa: E402
from flcourse.run import Run, build_federation, pick_malicious  # noqa: E402
from flcourse.server import apply_aggregate, sample_clients  # noqa: E402
from flcourse.training import evaluate  # noqa: E402


def main(config, overrides=None):
    cfg = load_config(config, overrides)
    run = Run(cfg)
    att = cfg.attack or {}
    byzantine = pick_malicious(cfg.fl.num_clients, int(att.get("num_byzantine", 0)), cfg.seed)
    fed = build_federation(cfg, run, malicious=byzantine)
    upload_mode = cfg.fl.upload

    # ---- data poisoning: -------------------------------------------------------------------
    # done once, on the local datasets of the Byzantine clients
    backdoor = att.get("backdoor") or {}
    backdoor_test = None
    for cid in byzantine:
        if att.get("data") == "label_flip":
            fed.clients[cid].data = label_flipping(fed.clients[cid].data)
        elif att.get("data") == "backdoor":
            fed.clients[cid].data = poison_backdoor(fed.clients[cid].data, backdoor.get("target_label", 0),
                                                    backdoor.get("fraction", 0.5), backdoor.get("size", 3), cfg.seed)

    # for one of the graph. This tests on the global test set how well the attack is doing
    if att.get("data") == "backdoor":
        backdoor_test = backdoor_testset(fed.test, backdoor.get("target_label", 0), backdoor.get("size", 3))

    # ----- directly attacking gradient/model -----------------------------------------
    # a python function which implementes the attack
    vector_attack = make_attack(att.get("vector"))
    # robustness guarantee: this many byzantine clients may we have at most
    f = int(att.get("f", len(byzantine)))

    # ----- setting up defences -----------------------------------------------------------------
    # aggregation rules
    pre_aggregators = make_pre_aggregators(cfg.pre_aggregators, f_default=f)
    aggregator = make_aggregator(cfg.aggregator, f_default=f)

    # some datastructure cleaning up....
    if (aggregator.needs_reference or type(aggregator).__name__ == "CenteredClipping") and upload_mode == "model":
        warnings.warn(f"{type(aggregator).__name__} compares update *directions*; use fl.upload: delta or gradient.")
    root_client = None
    if aggregator.needs_reference: # e.g. FLTrust needs a trusted reference dataset
        root_size = int(cfg.get_path("server.root_dataset_size", 100))      # carved out of the test set
        root_client = Client(-1, fed.test.subset(np.arange(root_size)), cfg.client, cfg.seed, run.device)
        fed.test = fed.test.subset(np.arange(root_size, len(fed.test)))

    # as in FedAvg, this randomness will only select the order in which clients are trained and who drops out,
    # but not the actual training. Note that certain DL optimizers use randomness as a natural component 
    # for regularisation!
    rng = np.random.default_rng(cfg.seed)


    global_vec = fed.global_vector
    run.log(f"attack: data={att.get('data')} vector={vector_attack} omniscient={att.get('omniscient', False)}   "
            f"aggregator: {aggregator}   pre-aggregators: {pre_aggregators}")

    for rnd in run.rounds():
        selected = sample_clients(cfg.fl.num_clients, cfg.fl.clients_per_round, rng)

        #  1. every client does local GD -----------------------------------------------------------------------------
        uploads = {}
        for cid in selected:
            client = fed.clients[cid]
            measure(global_vec, "download: global model", src="server", dst=cid)
            with timer("client: local update", who=cid):
                upload, stats = client.local_update(fed.model, global_vec, upload_mode)
            uploads[cid] = to_backend(upload, cfg.backend)
            run.log_client(rnd, cid, malicious=client.malicious, **stats)

        # 2. Another round only for the Byzantine clients colluding over a shared channel.  -------------------
        byz_selected = [cid for cid in selected if cid in fed.malicious]
        honest_selected = [cid for cid in selected if cid not in fed.malicious]
        if vector_attack is not None and byz_selected:
            channel = CollusionChannel()
            for cid in byz_selected:                           # publish what each attacker knows
                channel.post(cid, uploads[cid])
            if att.get("omniscient", False):                   # global eavesdroping
                for cid in honest_selected:
                    channel.post(cid, uploads[cid], record=False)
            with timer("attack: craft byzantine vector", who=byz_selected[0]):
                byzantine_vector = vector_attack(channel.values())   # 
            for cid in byz_selected:
                uploads[cid] = byzantine_vector
                # You can also just do client preprocessing
                # uploads[cid] = SignFlipping()([uploads[cid]])      #
                
        #  3. each client submits ----------------------------------------------------------------------------------
        for cid in selected:
            measure(uploads[cid], f"upload: {upload_mode}", src=cid, dst="server")

        #  4. server --------------------------------------------------------------------------------------------
        reference = None
        if root_client is not None:
            with timer("server: root update (FLTrust)"):
                reference, _ = root_client.local_update(fed.model, global_vec, upload_mode)
                reference = to_backend(reference, cfg.backend)
        with timer("server: aggregate"):
            X = stack([uploads[c] for c in selected])
            for pre in pre_aggregators:
                X = pre(X)
            aggregate = aggregator(X, weights=[fed.clients[c].num_samples for c in selected], reference=reference)
            global_vec = apply_aggregate(global_vec, aggregate, upload_mode, cfg.fl.server_lr)

        # distance of the aggregate to the mean of the honest uploads (what the attack maximises). This will be put into our figure.
        deviation = float("nan")
        if honest_selected:
            H, xp = as_matrix([uploads[c] for c in honest_selected])
            deviation = xp.item(xp.norm(xp.mean(H, axis=0) - to_backend(aggregate, cfg.backend)))


        # 5. eval------- -----------------------------------------------------------------------------------------
        if rnd % cfg.eval.every == 0 or rnd == cfg.fl.rounds:
            with timer("server: evaluate"):
                set_flat_params(fed.model, global_vec)
                loss, acc = evaluate(fed.model, fed.test, cfg.eval.batch_size, run.device)
                extra = {}
                if backdoor_test is not None:
                    extra["backdoor_asr"] = evaluate(fed.model, backdoor_test, cfg.eval.batch_size, run.device)[1]
            run.log_global(rnd, test_loss=loss, test_acc=acc, deviation=deviation, **extra)

    return run.finish(attack=str(vector_attack), data_attack=att.get("data"), aggregator=str(aggregator),
                      pre_aggregators=str(pre_aggregators), byzantine=sorted(byzantine))


if __name__ == "__main__":
    main(**parse_args(__doc__))
