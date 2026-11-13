"""Client-level differential privacy: DP-FedAvg with central, local or distributed noise.

    python experiments/dp_fedavg.py --config experiments/configs/dp_central_mnist.yaml

Config section ``dp``::

    dp:
      mode: central            # central | local | distributed
      clip: 1.0                # C: l2 bound of every client upload (= sensitivity)
      noise_multiplier: 1.0    # sigma
      delta: 1.0e-5
      colluders: 0             # distributed: clients that know (and subtract) their own noise

=============  ===========================  ==========================  =====================
mode           who adds noise               noise per upload            noise in the sum
=============  ===========================  ==========================  =====================
central        the (trusted) server         --                          N(0, sigma^2 C^2)
local          every client                 N(0, sigma^2 C^2)           N(0, m sigma^2 C^2)
distributed    every client (+ SecAgg)      N(0, sigma^2 C^2 / m)       N(0, sigma^2 C^2)
=============  ===========================  ==========================  =====================

``upload`` must be ``delta`` or ``gradient`` -- clipping a full model would destroy it.
Distributed DP only protects against the server if it cannot see individual uploads, i.e.
together with secure aggregation: see ``experiments/secure_aggregation.py`` (dp section).
The privacy loss epsilon is tracked with the RDP accountant and plotted per round.

Note that one can quantify what privacy definitions a specific added noise gives. In the lecture you use (eps,delta) DP and Renyi DP

"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from flcourse.backend import as_matrix, to_backend  # noqa: E402
from flcourse.bench import measure, timer  # noqa: E402
from flcourse.config import load_config, parse_args  # noqa: E402
from flcourse.dp import (RDPAccountant, add_gaussian_noise, clip_l2, effective_noise_multiplier,epsilon_after, l2_norm)
from flcourse.models import set_flat_params  # noqa: E402
from flcourse.run import Run, build_federation  # noqa: E402
from flcourse.server import apply_aggregate, num_selected, sample_clients  # noqa: E402
from flcourse.training import evaluate  # noqa: E402


def main(config, overrides=None):

    # -------- Some initialisation --------------------------------
    cfg = load_config(config, overrides)
    if cfg.fl.upload == "model":
        raise ValueError("DP clips the client's *change*: set fl.upload to 'delta' (or 'gradient').")
    dp = cfg.dp or {}
    mode = dp.get("mode", "central")
    C, sigma, delta = float(dp.get("clip", 1.0)), float(dp.get("noise_multiplier", 1.0)), float(dp.get("delta", 1e-5))
    colluders = int(dp.get("colluders", 0))
    run = Run(cfg)
    fed = build_federation(cfg, run)
    N = int(cfg.fl.num_clients)
    m = num_selected(N, cfg.fl.clients_per_round)          
    q = m / N

    # The Renyi DP evaluator. This will tell us, how private our noised model/gradient is.                                              
    accountant = RDPAccountant()

    
    participations = np.zeros(N, dtype=int)

    # We fix the DP noiser (Gaussian)
    noise_gen = torch.Generator().manual_seed(int(cfg.seed) + 7)

    # This randomness determines the client selection and NOT the gradient generation. Note that various GD optimizers use randomness as part of regularisation which is not fixed here! -> If you want that DIY :)
    rng = np.random.default_rng(cfg.seed)

    # Initialize init. grad.
    global_vec = fed.global_vector
    run.log(f"DP mode={mode}  clip C={C}  noise multiplier sigma={sigma}  delta={delta}  m={m}  q={q:.3f}")

    # --------------- FL rounds -----------------------------
    for rnd in run.rounds():
        selected = sample_clients(N, cfg.fl.clients_per_round, rng)

        uploads = []
        for cid in selected:
            client = fed.clients[cid]
            measure(global_vec, "download: global model", src="server", dst=cid)
            with timer("client: local update", who=cid):
                upload, stats = client.local_update(fed.model, global_vec, cfg.fl.upload)

            with timer("client: DP", who=cid):
                # diagnostic only, NOT part of the mechanism: the pre-clipping norm of every update is
                # logged (clients.csv + report notes: median, 90th percentile, fraction clipped), so you
                # can judge whether the clip C is far too tight or too loose for this model/learning rate
                norm = l2_norm(upload)
                upload = clip_l2(upload, max_norm=C)                           # sensitivity <= C
                if mode == "local":
                    upload = add_gaussian_noise(upload, std=sigma * C, generator=noise_gen)
                elif mode == "distributed":
                    upload = add_gaussian_noise(upload, std=sigma * C / math.sqrt(m), generator=noise_gen)

            upload = to_backend(upload, cfg.backend)
            measure(upload, f"upload: {cfg.fl.upload}", src=cid, dst="server")
            uploads.append(upload)
            run.log_client(rnd, cid, update_norm=norm, **stats)
        participations[selected] += 1

        with timer("server: aggregate"):
            X, xp = as_matrix(uploads)
            total = xp.sum(X, axis=0)
            if mode == "central":
                total = add_gaussian_noise(to_backend(total, "torch"), std=sigma * C, generator=noise_gen)
            aggregate = total / m                                             
            global_vec = apply_aggregate(global_vec, aggregate, cfg.fl.upload, cfg.fl.server_lr)

        # ---- privacy accounting -------
        if mode == "local":
            # every participation is a Gaussian mechanism on the client's own data: replacing the
            # client's data moves the clipped upload by <= 2C, the noise std is sigma*C -> multiplier sigma/2
            eps = epsilon_after(sigma / 2, 1.0, int(participations.max()), delta)
            #eps = epsilon_after(0, 1.0, 10, delta)
        else:
            sigma_eff = sigma if mode == "central" else effective_noise_multiplier(sigma, m, colluders)
            accountant.step(sigma_eff, q)
            eps = accountant.get_epsilon(delta)

        # eval
        if rnd % cfg.eval.every == 0 or rnd == cfg.fl.rounds:
            with timer("server: evaluate"):
                set_flat_params(fed.model, global_vec)
                loss, acc = evaluate(fed.model, fed.test, cfg.eval.batch_size, run.device)
            run.log_global(rnd, test_loss=loss, test_acc=acc, epsilon=eps)

    norms = [r["update_norm"] for r in run.client_rows]
    notes = (f"DP: mode={mode}, C={C}, sigma={sigma}, delta={delta}, cohort m={m} of N={N} (q={q:.3f})\n"
        f"final epsilon = {eps:.3f}  (({eps:.2f}, {delta:g})-DP at the client level"
        f"{', per client (local DP)' if mode == 'local' else ''})\n"
        f"client update norms before clipping: median {np.median(norms):.3f}, "
        f"90th percentile {np.percentile(norms, 90):.3f}  -> fraction clipped "
        f"{np.mean(np.array(norms) > C):.2f}")
    return run.finish(notes=notes, epsilon=eps, dp_mode=mode)


if __name__ == "__main__":
    main(**parse_args(__doc__))
