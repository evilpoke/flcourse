"""Federated learning with secure aggregation: SecAgg, SecAgg+ or LightSecAgg.

    python experiments/secure_aggregation.py --config experiments/configs/secaggplus_mnist.yaml

The client loop is the same as in fedavg.py. Instead of ``aggregator(stack(uploads))``
the server runs a secure-aggregation protocol: inside ``secagg.aggregate(...)`` every
protocol stage is one loop over the clients, and all key material, secret shares and
masked vectors are recorded with ``measure`` (see report.txt / communication.png).

Config section ``secagg``:

    secagg:
      protocol: secagg+             # secagg | secagg+ | lightsecagg
      num_shares: 5                 # secagg+: neighbours per client (incl. itself)
      reconstruction_threshold: 3   # secagg/secagg+: Shamir threshold t
      clipping_range: 1.0           # inputs are clipped to [-c, c], then quantised
      quantization_range: 4194304   # ... to integers in [0, 2^22]
      dropout_rate: 0.1             # fraction of the selected clients that drop out ...
      dropout_stage: masked_input   # ... before this stage (share_keys | masked_input | unmask)
      recovery_dropout_rate: 0.2    # lightsecagg only, 2nd dropout filter: fraction of the announced
                                    # survivors U_1 that never send their aggregated encoded mask

An optional ``dp`` section with ``mode: distributed`` adds distributed DP: every client
adds its share of the Gaussian noise *before* masking, so the server only ever sees the
noisy sum.

Robust aggregation is not available here on purpose: the server only learns the SUM of
the inputs, so it cannot compute medians, distances (Krum) or trust scores (FLTrust).
Combining robustness with input privacy needs dedicated protocols (see README).
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from flcourse.backend import to_numpy  # noqa: E402
from flcourse.bench import measure, timer  # noqa: E402
from flcourse.config import load_config, parse_args  # noqa: E402
from flcourse.dp import RDPAccountant, add_gaussian_noise, clip_l2  # noqa: E402
from flcourse.models import set_flat_params  # noqa: E402
from flcourse.run import Run, build_federation  # noqa: E402
from flcourse.secagg import make_secure_aggregator  # noqa: E402
from flcourse.server import apply_aggregate, num_selected, sample_clients  # noqa: E402
from flcourse.training import evaluate  # noqa: E402

ROBUSTNESS_VS_PRIVACY = (
    "Secure aggregation reveals only the SUM of the client inputs, so the server cannot run "
    "'{name}' (it needs every individual update). Use aggregator: mean here. Robust *and* private "
    "aggregation needs dedicated protocols (e.g. norm-bound proofs as in RoFL/EIFFeL, robust "
    "aggregation inside MPC as in BREA/ELSA) -- see the README section 'Robustness + privacy'.")


def main(config, overrides=None):
    cfg = load_config(config, overrides)
    agg_name = str((cfg.aggregator or {}).get("name", "mean")).lower()
    if agg_name not in ("mean", "average"):
        raise ValueError(ROBUSTNESS_VS_PRIVACY.format(name=agg_name))
    run = Run(cfg)
    fed = build_federation(cfg, run)
    sa_cfg = cfg.secagg or {"protocol": "secagg"}
    secagg = make_secure_aggregator(sa_cfg, seed=int(cfg.seed))


    dropout_rate = float(sa_cfg.get("dropout_rate", 0.0)) # Per FL communication round, we randomly sample dropouts. All (slected) clients however recieve the new global model in the beginning of the next FL communication round.
    dropout_stage = sa_cfg.get("dropout_stage", "masked_input") # Denotes the stage *before which* dropouts happen. As for SecAgg, we set this to the masked_input stage where a local model/gradient is masked with the pairwise mask and uploaded (or not in case of dropouts). The interesting thing with SecAgg is then, that with the secret sharing of the private key material, the contributed material of the other parties can still be used (see SecAgg lectures).
    recovery_dropout_rate = float(sa_cfg.get("recovery_dropout_rate", 0.0))  # LightSecAgg: 2nd dropout filter (step 3)
    if recovery_dropout_rate and secagg.name != "lightsecagg":
        raise ValueError("secagg.recovery_dropout_rate is only defined for protocol: lightsecagg")


    dp = cfg.dp or {}
    m = num_selected(int(cfg.fl.num_clients), cfg.fl.clients_per_round)
    accountant = RDPAccountant() if dp else None
    noise_gen = torch.Generator().manual_seed(int(cfg.seed) + 7)

    # randomness only controls client selection and NOT the DL optimizer which include randomness
    rng = np.random.default_rng(cfg.seed)

    # initial model/gradient
    global_vec = fed.global_vector
    run.log(f"secure aggregation: {secagg}   dropouts: {dropout_rate:.0%} before '{dropout_stage}'"
            + (f", then {recovery_dropout_rate:.0%} of U_1 before 'recovery'" if recovery_dropout_rate else "")
            + (f"   distributed DP: C={dp.get('clip')} sigma={dp.get('noise_multiplier')}" if dp else ""))

    for rnd in run.rounds():
        # note that not all clients may be considered. Of the selected onces, we then have dropouts 
        selected = sample_clients(cfg.fl.num_clients, cfg.fl.clients_per_round, rng)
        
        # ---- 1. clients ----------------------------------------------------------------------
        # Distribution of the current global model
        uploads = {}
        for cid in selected:
            client = fed.clients[cid]
            measure(global_vec, "download: global model", src="server", dst=cid)
            with timer("client: local update", who=cid):
                upload, stats = client.local_update(fed.model, global_vec, cfg.fl.upload)

            # ---- 2. client-side post-processing --------------------------------------------------
            # we may add noise locally before applying masks
            if dp:                                         # distributed DP: clip + this client's noise share
                upload = clip_l2(upload, max_norm=float(dp["clip"]))
                upload = add_gaussian_noise(upload, std=float(dp["noise_multiplier"]) * float(dp["clip"])
                                            / math.sqrt(m), generator=noise_gen)

            uploads[cid] = to_numpy(upload)                # the protocols work on numpy arrays
            run.log_client(rnd, cid, **stats)

        # ---- 3. some clients go offline during the protocol ---------------------------------------
        num_drop = int(round(dropout_rate * len(selected)))
        dropped = rng.choice(selected, size=num_drop, replace=False).tolist() if num_drop else []
        dropouts = {cid: dropout_stage for cid in dropped}
        if recovery_dropout_rate:
            # LightSecAgg, 2nd filter: the server announces the survivors U_1 (= clients that uploaded a
            # masked model) and every member of U_1 should send its aggregated encoded mask -- but some of
            # them go offline before doing so. The server can decode from any U of these messages.
            announced = [cid for cid in selected if cid not in dropouts]            # U_1
            num_drop = int(round(recovery_dropout_rate * len(announced)))
            dropped = rng.choice(announced, size=num_drop, replace=False).tolist() if num_drop else []
            dropouts.update({cid: "recovery" for cid in dropped})

        # ---- 4. secure aggregation: the server learns only the mean of the survivors ---------------
        aggregate = secagg.aggregate(uploads, dropouts=dropouts)       # all protocol stages, measured
        global_vec = apply_aggregate(global_vec, aggregate, cfg.fl.upload, cfg.fl.server_lr)
        
        # sanity check (only possible in a simulation): compare with the plain mean
        survivors = secagg.last_info["aggregated_clients"]
        plain = np.mean([uploads[c] for c in survivors], axis=0)
        error = float(np.max(np.abs(plain - aggregate)))
        if dp:
            accountant.step(float(dp["noise_multiplier"]) * math.sqrt(len(survivors) / m), m / cfg.fl.num_clients)

        # ---- 5. evaluation --------------------------------------------------------------------------
        if rnd % cfg.eval.every == 0 or rnd == cfg.fl.rounds:
            with timer("server: evaluate"):
                set_flat_params(fed.model, global_vec)
                loss, acc = evaluate(fed.model, fed.test, cfg.eval.batch_size, run.device)
            extra = {"epsilon": accountant.get_epsilon(float(dp.get("delta", 1e-5)))} if dp else {}
            if "responses" in secagg.last_info:      # LightSecAgg: aggregated encoded masks that arrived
                extra["recovery_responses"] = secagg.last_info["responses"]
            run.log_global(rnd, test_loss=loss, test_acc=acc, survivors=len(survivors),
                           secagg_max_error=error, **extra)

    info = {k: v for k, v in secagg.last_info.items() if k != "aggregated_clients"}
    notes = (f"parameters: {secagg}\nlast round: {info}\n"
             f"max |secure mean - plain mean| over the last round: {error:.2e} "
             f"(quantisation step = {2 * secagg.params.clipping_range / secagg.params.quantization_range:.1e})")
    return run.finish(notes=notes, protocol=secagg.name)


if __name__ == "__main__":
    main(**parse_args(__doc__))
