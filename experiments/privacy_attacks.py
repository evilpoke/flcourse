"""Gradient inversion by an honest-but-curious server(=federator): 
    - The server looks at either the individual upload or the aggregate. 
    Note that even the aggregate may hold sensitive data in an extractable way
    - The individual gradients are either not protected, or protected with local DP or central DP
    Note that for central DP you need a trusted aggregator in your threat model
    
    Run:
        
        python experiments/privacy_attacks.py --config experiments/configs/privacy_attacks_mnist.yaml
        python experiments/privacy_attacks.py --config ... --set privacy.dp_mode=central

Every participating client computes the gradient of ONE private image (FedSGD, batch
size 1). The server then attacks the aggregate or the individual uploads.

* the INDIVIDUAL upload of ``privacy.target_client`` (DLG + iDLG label trick);
* the AGGREGATE: the mean over the n participants' gradients. All clients compute at the
  same global model, so this mean equals the gradient of one batch of the n images and
  batch-DLG applies: the attacker optimises n dummy images at once (dummy i with the label
  of participant i -- all labels are known to the attacker) until the mean of their
  gradients matches the aggregate. No participant is special here: ``aggregate_all_clients.png``
  shows this whole set for every setting, each dummy under its participant.
  (``gradient_inversion.png`` shows the attack on the target's individual upload.)

``privacy.dp_mode`` decides what the DP mechanism changes (C = ``privacy.dp_clip``, one
column per sigma in ``privacy.dp_noise_multipliers``):

==========  =====================================  ==========================================
dp_mode     individual upload the server HOLDS      aggregate the server computes / releases
==========  =====================================  ==========================================
none        raw gradient                            mean of the raw gradients
local       clip(C) + N(0, (sigma C)^2 I)           mean of the noised uploads -> their noise
                                                    averages down to std sigma C / sqrt(n)
central     clip(C) only! the server adds its       mean of the clipped uploads
            noise AFTER seeing the uploads          + N(0, (sigma C / n)^2 I)
==========  =====================================  ==========================================

The two punchlines:

* ``central``: the individual row does not change with sigma. Central DP protects the
  *released aggregate* (and thus the model) -- it protects nothing from the server
  itself, which sees the raw uploads before adding noise.
* ``local``: the sigma that destroys the individual reconstruction is diluted by
  averaging in the aggregate (sigma / sqrt(n) vs. the central sigma / n) -- the price of
  not trusting the server, and why local DP costs more utility at the same noise level.

Each noised panel is captioned with the single-release (epsilon, delta)-guarantee of its
mechanism (same accounting as dp_fedavg.py; delta = ``privacy.delta``). Note how sigmas
that visibly wreck the image can still carry astronomically large epsilon: "looks
private" is not a privacy guarantee.

Secure aggregation is on purpose NOT a switch here: it changes who sees what (the server
then holds no individual upload unless >= threshold clients collude with it -- see the
README), not the values of the aggregate, so it would not change any panel.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from flcourse.attacks import infer_label_idlg, invert_gradient  # noqa: E402
from flcourse.backend import stack  # noqa: E402
from flcourse.bench import measure, timer  # noqa: E402
from flcourse.config import load_config, parse_args  # noqa: E402
from flcourse.dp import add_gaussian_noise, clip_l2, epsilon_after  # noqa: E402
from flcourse.run import Run, build_federation  # noqa: E402
from flcourse.server import num_selected  # noqa: E402
from flcourse.training import compute_gradient  # noqa: E402

MODES = ("none", "local", "central")


def to_pixels(x: torch.Tensor, data) -> np.ndarray:
    """Normalised image (C, H, W) -> H x W (x C) array in [0, 1] for plotting."""
    if data.mean is not None:
        mean = torch.tensor(data.mean).view(-1, 1, 1).to(x.device)
        std = torch.tensor(data.std).view(-1, 1, 1).to(x.device)
        x = x * std + mean
    img = x.detach().clamp(0, 1).cpu().numpy()
    return img[0] if img.shape[0] == 1 else img.transpose(1, 2, 0)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return float(10 * np.log10(1.0 / max(mse, 1e-12)))


def eps_caption(noise_multiplier: float, delta: float) -> str:
    """Single-release (epsilon, delta) of a Gaussian mechanism with this noise multiplier."""
    eps = epsilon_after(noise_multiplier, 1.0, 1, delta)
    return f"eps = {eps:.3g}" if eps < 1e4 else f"eps > 1e4 (!)"


@dataclass
class Panel:
    """One image of the result figures: a reconstruction of the server in one setting."""
    setting: str          # "no DP" or "sigma=0.01" -- also the key in report.txt
    caption: str          # panel title: the setting and the privacy guarantee of what was attacked
    image: np.ndarray     # the reconstruction, H x W (x C) in [0, 1]
    psnr: float           # similarity to the private image it tries to recover, in dB (> ~25 dB = recognisable)


def main(config, overrides=None):
    
    # loading configs
    cfg = load_config(config, overrides)
    pa = cfg.privacy
    mode = str(pa.get("dp_mode", "none")).lower()
    if mode not in MODES:
        raise ValueError(f"privacy.dp_mode must be one of {MODES}, got {mode!r}")
    C, delta = float(pa.dp_clip), float(pa.get("delta", 1e-5))
    sigmas = [float(s) for s in pa.dp_noise_multipliers] if mode != "none" else []
    
    # 
    run = Run(cfg)
    fed = build_federation(cfg, run)
    model, target = fed.model, int(pa.target_client)
    # the round's participants are simply the first clients_per_round clients (deterministic)
    participants = list(range(num_selected(int(cfg.fl.num_clients), cfg.fl.clients_per_round)))
    if target not in participants:
        raise ValueError(f"privacy.target_client={target} is not among the participants {participants}")
    noise_gen = torch.Generator().manual_seed(int(cfg.seed) + 7)

    # ---- every participant computes the gradient of ONE private image --------------------------
    grads, labels, private_images = {}, {}, {}
    for cid in participants:
        one = fed.clients[cid].data.subset([0])
        labels[cid] = int(one.y[0])
        private_images[cid] = to_pixels(one.batch(torch.arange(1), run.device)[0][0], fed.train)
        with timer("client: gradient", who=cid):
            grads[cid], _ = compute_gradient(model, one, batch_size=1, num_batches=1, device=run.device)
        measure(grads[cid], "upload: gradient", src=cid, dst="server")
    n = len(participants)
    true_img = private_images[target]
    run.log(f"dp_mode={mode}  n={n} participants  C={C:g}  target=client {target} (label {labels[target]})")

    def average(uploads):                                # the aggregate: mean of the n participants' uploads
        return torch.mean(stack(list(uploads.values())), dim=0)

    # ---- the two attacks: each returns one Panel of the figure ----------------------------------
    def attack_individual(upload, setting, caption):
        label = infer_label_idlg(model, upload)          # iDLG: read the label off the gradient
        with timer(f"server: invert individual ({setting})"):
            x_rec, _, _ = invert_gradient(model, upload, fed.train.input_shape, fed.train.num_classes,
                                          label=label, iterations=int(pa.iterations), seed=int(cfg.seed))
        image = to_pixels(x_rec[0], fed.train)
        run.log(f"  individual | {setting:<12} -> PSNR {psnr(image, true_img):5.1f} dB")
        return Panel(setting, caption, image, psnr(image, true_img))

    def attack_aggregate(aggregate, setting, caption):
        """Batch-DLG: all clients computed their gradient at the same weights, so the mean of the n
        batch-1 gradients IS the gradient of one batch of the n images. The attacker optimises n dummy
        images at once (dummy i with participant i's label -- all labels are known to the attacker)
        until the mean of their gradients matches the aggregate.
        Returns the whole reconstructed set, one dummy per participant: {client id: Panel}."""
        with timer(f"server: invert aggregate ({setting})"):
            x_rec, _, _ = invert_gradient(model, aggregate, fed.train.input_shape, fed.train.num_classes,
                                          label=[labels[cid] for cid in participants], batch_size=n,
                                          iterations=int(pa.iterations), seed=int(cfg.seed))
        dummies = [to_pixels(x_rec[d], fed.train) for d in range(n)]
        # Participants with the same label are interchangeable for the attacker (the dummies of two "4"s
        # can come out in either order). For the figure, pair every participant with one dummy of ITS
        # label such that the total PSNR is highest (Hungarian algorithm; other labels are forbidden).
        # This uses the private images: it only orders the attacker's set, the attacker cannot do it.
        score = np.array([[psnr(dummies[d], private_images[cid]) if labels[participants[d]] == labels[cid]
                           else -1e9 for d in range(n)] for cid in participants])
        _, dummy_of = linear_sum_assignment(score, maximize=True)     # dummy_of[i]: participant i's dummy
        panels = {cid: Panel(setting, caption, dummies[d], psnr(dummies[d], private_images[cid]))
                  for cid, d in zip(participants, dummy_of)}
        run.log(f"  aggregate  | {setting:<12} -> PSNR per participant: "
                + "  ".join(f"{panel.psnr:.1f}" for panel in panels.values()))
        return panels

    ##---------- Now testing the inversion attacks -------------

    # Every setting is one column of the figure: "no DP", then one column per noise multiplier sigma
    # (noise std = sigma * C). In each setting all participants upload, and the server attacks the
    # target's individual upload (gradient_inversion.png) and the aggregate of all n uploads. The attack
    # on the aggregate returns a whole set of n images: one row of aggregate_all_clients.png per setting.
    individual_row = []                                  # one Panel per setting
    aggregate_rows = []                                  # one {client id: Panel} per setting

    # attacking gradients without DP protection
    uploads = grads                                      # every client uploads its raw gradient
    aggregate = average(uploads)
    individual_row.append(attack_individual(uploads[target], "no DP", caption="no DP"))
    aggregate_rows.append(attack_aggregate(aggregate, "no DP", caption="no DP"))

    # attacking with DP protection
    # We test for all DP sigmas how well
    for sigma in sigmas:
        setting = f"sigma={sigma:g}"
        if mode == "local":
            # every client clips AND noises its own gradient; the server only averages the uploads
            uploads = {cid: add_gaussian_noise(clip_l2(g, C), std=sigma * C, generator=noise_gen)
                       for cid, g in grads.items()}
            aggregate = average(uploads)
            # an upload is a Gaussian mechanism on one client's data: replacing that data moves the
            # clipped gradient by <= 2C, the noise std is sigma*C -> noise multiplier sigma/2 (as in dp_fedavg)
            caption_individual = f"{setting}  ({eps_caption(sigma / 2, delta)})"
            # the aggregate is computed from these private uploads alone: post-processing, no new privacy loss
            caption_aggregate = f"{setting}  (same eps:\npost-processing)"
        elif mode == "central":
            # every client only clips; the server averages and THEN adds noise: (sum + N(0, (sigma C)^2)) / n
            uploads = {cid: clip_l2(g, C) for cid, g in grads.items()}
            aggregate = add_gaussian_noise(average(uploads), std=sigma * C / n, generator=noise_gen)
            # the server holds the upload before it adds any noise: the same reconstruction for every sigma
            caption_individual = f"{setting}\nunchanged: server noise\ncomes after the upload"
            # one client changes the clipped sum by <= C, the noise std is sigma*C -> noise multiplier sigma
            caption_aggregate = f"{setting}  ({eps_caption(sigma, delta)})"
        individual_row.append(attack_individual(uploads[target], setting, caption_individual))
        aggregate_rows.append(attack_aggregate(aggregate, setting, caption_aggregate))

    # ---- figure 1: the target's individual upload, one column per setting -------------------------
    ncols = 1 + len(individual_row)                      # column 0: the target's private image
    fig, axes = plt.subplots(1, ncols, figsize=(2.3 * ncols, 3.0), squeeze=False)
    axes = axes[0]
    for ax in axes:
        ax.set_xticks([]), ax.set_yticks([])
    axes[0].imshow(true_img, cmap="gray" if true_img.ndim == 2 else None, vmin=0, vmax=1)
    axes[0].set_title("private image\nof the target", fontsize=8)
    axes[1].set_ylabel("individual upload", fontsize=9)
    for i, panel in enumerate(individual_row):
        axes[1 + i].imshow(panel.image, cmap="gray" if panel.image.ndim == 2 else None, vmin=0, vmax=1)
        axes[1 + i].set_title(f"{panel.caption}\nPSNR {panel.psnr:.1f} dB", fontsize=7)
    fig.suptitle(f"Gradient inversion (DLG/iDLG) of the target's upload under dp_mode = {mode}"
                 + (f"   (C = {C:g}, delta = {delta:g})" if mode != "none" else ""), fontsize=10)
    fig.tight_layout()
    fig.savefig(run.out_dir / "gradient_inversion.png", dpi=150)
    plt.close(fig)

    # ---- figure 2: the aggregate inverted for every participant (row 0: their private images) ----
    fig, axes = plt.subplots(1 + len(aggregate_rows), n, figsize=(1.5 * n, 1.75 * (1 + len(aggregate_rows))),
                             squeeze=False)
    for ax in axes.flat:
        ax.set_xticks([]), ax.set_yticks([])
    for j, cid in enumerate(participants):
        axes[0, j].imshow(private_images[cid], cmap="gray" if true_img.ndim == 2 else None, vmin=0, vmax=1)
        axes[0, j].set_title(f"client {cid}\nlabel {labels[cid]}", fontsize=7)
        for r, row in enumerate(aggregate_rows, start=1):
            axes[r, j].imshow(row[cid].image, cmap="gray" if true_img.ndim == 2 else None, vmin=0, vmax=1)
            axes[r, j].set_title(f"PSNR {row[cid].psnr:.1f} dB", fontsize=7)
    axes[0, 0].set_ylabel("private image", fontsize=8)
    for r, row in enumerate(aggregate_rows, start=1):
        axes[r, 0].set_ylabel(row[participants[0]].caption, fontsize=7)   # one caption per row
    fig.suptitle(f"Joint inversion of the aggregate (mean of n = {n} gradients): the set of n images the "
                 f"optimiser returns, one row per setting, dp_mode = {mode}\neach dummy under the participant "
                 "with its label (same-label dummies paired by best match); PSNR vs. that participant's image",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(run.out_dir / "aggregate_all_clients.png", dpi=150)
    plt.close(fig)

    # ---- report: the PSNR of every panel ------------------------------------------------------------
    psnr_individual = {panel.setting: panel.psnr for panel in individual_row}
    psnr_aggregate = {row[participants[0]].setting: {cid: row[cid].psnr for cid in participants}
                      for row in aggregate_rows}
    lines = ["individual upload: PSNR of the reconstruction vs. the target's private image "
             "(> ~25 dB = recognisable)"]
    lines += [f"  {k:<12} {v:6.1f} dB" for k, v in psnr_individual.items()]
    lines += ["", "joint inversion of the aggregate: PSNR of every participant's dummy vs. its own image (dB)",
              "  setting      " + "".join(f"{f'c{cid} ({labels[cid]})':>9}" for cid in participants)]
    lines += [f"  {setting:<12} " + "".join(f"{p:9.1f}" for p in row.values())
              for setting, row in psnr_aggregate.items()]
    notes = "\n".join(lines) + "\n\nfigures: gradient_inversion.png, aggregate_all_clients.png"
    return run.finish(notes=notes, dp_mode=mode, psnr_individual=psnr_individual, psnr_aggregate=psnr_aggregate)


if __name__ == "__main__":
    main(**parse_args(__doc__))
