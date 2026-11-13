# flcourse — a readable federated learning framework for teaching

Important notes straight ahead: Use only in the context of the TDL course. In that context, take note of two things which make this codebase unusable in a real-world setting. 

-  We do not parallelise the FL clients! 
This menas that the experiments in `experiments/*.py` typically have a `for` loop over all clients
- Clients do not "own" a network. One shared "worker" model is overwritten in-place with the
global weights of another client (see `Client.local_update`), so memory does not grow with the
number of clients. All uploads are flat vectors (`get_flat_params` / `set_flat_params`).

To make the implementation and rapid prototyping as pleasant as possible, note these quality-of-live features:

- aggregators, pre-aggregators and attacks are callables on a stack of vectors, and accept **NumPy arrays and PyTorch tensors** 
- upload material may be either model or gradients: `fl.upload: model | delta | gradient`
- **attacks**: sign flipping, IPM, ALIE, Gaussian, Mimic, Inf, Zero, Scaling (model replacement),
  Min-Max/Min-Sum, label flipping, backdoor triggers. Colluding clients share a `CollusionChannel` in a separate loop in script `byzantine.py`
- **Robustness**: Mean, weighted FedAvg, Median, Trimmed mean, Geometric median (RFA),
  Krum, Multi-Krum, Bulyan, Centered clipping, FLTrust; pre-aggregators NNM, Bucketing,
  Clipping, ARC.
- **DP**: experiments in `dp_fedavg.py` for benchmarking and inversion attacks in `privacy_attacks.py`. We use Gaussian noise
- Secure aggregation: SecAgg, Secagg+ and LightSecAgg
- **Benchmarking**: `measure(x)` gives the bits needed to serialise any object (tensors, keys,
  shares, ciphertexts, ...) and `timer(label)` times any step.
- **output** is: `report.txt`, `metrics.json`, `clients.csv`, `config.yaml` and some figures


## Install

1. Install conda
2. Setup a conda environment for 3.12.16
3. Activate that environment
4. Do:

```
pip install -r requirements.txt                         # or: pip install -e .
python experiments/download_data.py {parameters}        # (see -h for parameters)
python -m pytest -q                        # 116 tests, ~45 s on a laptop CPU
```

## Use

As I mentioned in the tutorial Nov. 12th there are basically four python scripts each with the double-nested training loop (explained in the tutorial session and again outlined later in this README)

- `fedavg.py`:  The normal FL communication loop.
- `byzantine.py`: The FL loop is complemented by another loop afterwards, that changes some gradients/models with the help of a collusion channel. Also there is dataset poisoning
- `dp_fedavg.py`: The clients add DP noise
- `secure_aggregation.py`: The clients hide their model with the SegAgg mask. Note that SecAgg is a generalisation of SecAgg+ (see lecture). I also added LightSecAgg as a more lightweight alternative fo SecAgg along with not having to rely on the Computational Diffie Hellman assumption. If that confuses you, ask yourself: Is SecAgg (Bonawitz et al., CCS 2017) perfectly secure? 

Each of these scripts requires a config YAML. Have a look into the `experiments/configs` for self-explanatory examples.   
To run e.g. FedAvg, just to

    python experiments/fedavg.py --config experiments/configs/quick_digits.yaml 

## Use GPUS

In case you want to use GPUs, you can do so. Pytorchs should detect the GPU. 
Note that with colab, you have (somewhat limited) access to GPUs, but it only works with jupyter notebooks. To make you life as easy as possible, I have in `notebooks/colab_quickstart.ipynb` a notebook that jumpstarts a run.   
Just upload that notebook and things should be obvious from there.

## Repository layout

```
flcourse/                    the library (no training loops in here)
  backend.py                 NumPy/PyTorch compatibility layer (as_matrix, xp.median, ...)
  config.py                  YAML configs with defaults, --set overrides
  device.py                  robust cpu/cuda/mps selection, seeding
  data/datasets.py           mnist, fashion_mnist, cifar10, digits (offline), synthetic (offline)
  data/partition.py          iid, dirichlet(alpha), shards
  models.py                  logreg, mlp, lenet5, cnn, cnn_large, resnet18, dlg_lenet
  training.py                local_sgd, compute_gradient, evaluate
  client.py                  Client.local_update(model, global_vector, upload)
  server.py                  sample_clients, apply_aggregate (model/delta/gradient)
  aggregators.py             Average ... FLTrust  (ByzFL interface)
  preaggregators.py          NNM, Bucketing, Clipping, ARC
  attacks/                   model_poisoning.py, data_poisoning.py, collusion.py, gradient_inversion.py
  dp/                        mechanisms.py (clip, noise), accountant.py (RDP)
  secagg/flwr/               primitives vendored from Flower (Apache-2.0)
  secagg/secaggplus.py       SecAgg / SecAgg+ as stage-by-stage loops (Flower's logic and checks)
  secagg/lightsecagg.py      LightSecAgg (masks coded with a T-private MDS matrix over F_p, via galois)
  bench.py                   measure(), timer(), Bench (tables, latency estimate)
  report.py                  convergence / communication / computation / partition plots
  run.py                     Run (output folder, metrics, report), build_federation

experiments/                 one file per experiment, all called with --config
  fedavg.py                  the reference loop (FedAvg / FedSGD)
  byzantine.py               + data/model poisoning, collusion channel, robust aggregation
  dp_fedavg.py               + client-level DP (central | local | distributed) and accounting
  secure_aggregation.py      + SecAgg / SecAgg+ / LightSecAgg (with dropouts, optional distributed DP)
  secagg_scaling.py          protocol cost vs. number of clients
  privacy_attacks.py         gradient inversion (DLG/iDLG): individual upload vs. aggregate,
                             under no / local / central DP
  sweep.py                   grid of runs -> heatmap (e.g. attacks x aggregators)
  download_data.py           pre-download the datasets into ./data
  configs/*.yaml             All config yamls. Just have a look into some of them. Should be self-explanatory

tests/                       pytest suite (unit + end-to-end)
notebooks/colab_quickstart.ipynb    -> To jumpstart a run in colab
```

## The training loop

This is `experiments/fedavg.py`, unabridged apart from imports. All other experiments are
variations of it.

```python
def main(config, overrides=None):
    cfg = load_config(config, overrides)
    run = Run(cfg)                                    # output folder, seeds, device, benchmark recorder
    fed = build_federation(cfg, run)                  # data -> partition -> clients; model
    pre_aggregators = make_pre_aggregators(cfg.pre_aggregators)
    aggregator = make_aggregator(cfg.aggregator)
    rng = np.random.default_rng(cfg.seed)
    upload_mode = cfg.fl.upload                       # "model" | "delta" | "gradient"
    global_vec = fed.global_vector                    # the global model as one flat torch vector

    for rnd in run.rounds():
        selected = sample_clients(cfg.fl.num_clients, cfg.fl.clients_per_round, rng)

        # ---- 1. clients ----------------------------------------------------------
        uploads = []
        for cid in selected:
            client = fed.clients[cid]
            measure(global_vec, "download: global model", src="server", dst=cid)
            with timer("client: local update", who=cid):
                upload, stats = client.local_update(fed.model, global_vec, upload_mode)
            upload = to_backend(upload, cfg.backend)          # torch tensor (default) or numpy array

            # ---- 2. client-side post-processing (one line each; comment in / out) ---
            # upload = clip_l2(upload, max_norm=1.0)                      # norm bounding
            # upload = add_gaussian_noise(upload, std=0.01)               # (local) noise

            measure(upload, f"upload: {upload_mode}", src=cid, dst="server")
            uploads.append(upload)
            run.log_client(rnd, cid, **stats)

        # ---- 3. server: (pre-)aggregation and global update ----------------------
        with timer("server: aggregate"):
            X = stack(uploads)                                    # (n_selected, d) matrix
            for pre in pre_aggregators:                           # e.g. NNM, Bucketing, Clipping
                X = pre(X)
            weights = [fed.clients[c].num_samples for c in selected]
            aggregate = aggregator(X, weights=weights)            # e.g. mean, median, krum
            global_vec = apply_aggregate(global_vec, aggregate, upload_mode, cfg.fl.server_lr)

        # ---- 4. evaluate the global model ------------------------------------------
        if rnd % cfg.eval.every == 0 or rnd == cfg.fl.rounds:
            with timer("server: evaluate"):
                set_flat_params(fed.model, global_vec)
                loss, acc = evaluate(fed.model, fed.test, cfg.eval.batch_size, run.device)
            run.log_global(rnd, test_loss=loss, test_acc=acc)

    return run.finish()
```



## Model vs. gradient uploads


`model` and `delta` give identical results for the mean. With one full-batch local step and
`server_lr = lr`, all three modes coincide; `tests/test_training.py` checks this.
Use `delta` or `gradient` whenever an operation needs the *direction* of a client's change: DP
clipping, FLTrust, centered clipping, norm bounding, sign flipping.   

Take note of the *semantic* meaning of what you are uploading: Clipping a full model
would destroy it. `dp_fedavg.py` refuses `model` uploads for this reason.

## Post-processing stuff

Anything you want to do to an upload is one line of the form `upload = f(upload, ...)`.

There are no hooks or wrapper classes. All functions work on NumPy arrays and torch tensors.

```python
upload = clip_l2(upload, max_norm=1.0)                      # flcourse.dp
upload = add_gaussian_noise(upload, std=0.1)                # flcourse.dp
upload = SignFlipping()([upload])                           # flcourse.attacks (ByzFL interface)
upload = Scaling(factor=10.0)([upload])                     # model replacement / boosting
upload = to_backend(upload, "numpy")                        # switch library
```



## Aggregators, pre-aggregators, attacks (NumPy + PyTorch)

The ByzFL interface: an object with parameters, called on an `(n, d)` matrix or a list of
vectors. The result has the type of the input (NumPy in, NumPy out; torch in, torch out on
the same device).


| aggregator (`aggregator.name`) | reference | needs |
|---|---|---|
| `mean` | — | |
| `weighted_mean` / `fedavg` | McMahan et al. 2017 | `weights` (passed by the loop) |
| `median` | Yin et al. 2018 | |
| `trimmed_mean` (`f`) | Yin et al. 2018 | n > 2f |
| `geometric_median` / `rfa` | Pillutla et al. 2022 | |
| `krum`, `multi_krum` (`f`, `m`) | Blanchard et al. 2017 | n ≥ 2f+3 |
| `bulyan` (`f`) | El Mhamdi et al. 2018 | n ≥ 4f+3 |
| `centered_clipping` (`tau`, `L`) | Karimireddy et al. 2021 | stateful |
| `fltrust` | Cao et al. 2021 | server root data (`server.root_dataset_size`) |

Pre-aggregators (`pre_aggregators: [...]`): `nnm` (Allouah et al. 2023), `bucketing`
(Karimireddy et al. 2022), `clipping` (static l2), `arc` (Allouah et al. 2024).
If `f` is not given, the experiments use the number of Byzantine clients.

Model-poisoning attacks (`attack.vector`), each `attack(honest_vectors) -> vector`: `sign_flipping`,
`ipm` (Xie et al. 2020), `alie` (Baruch et al. 2019), `gaussian`, `mimic`, `inf`, `zero`,
`scaling` (Bagdasaryan et al. 2020), `minmax` / `minsum` (Shejwalkar & Houmansadr 2021).
Data poisoning (`attack.data`): `label_flip`, `backdoor` (BadNets trigger, attack success
rate logged as `backdoor_asr`). Privacy: `invert_gradient` (DLG + iDLG label trick).


## Benchmarking: `measure` and `timer`

In your project you may want/need to do benchmarking for communication and computation overhead. 


```python
from flcourse.bench import measure, timer
bits = measure(gradient)                                         # just the size in bits
measure(gradient, "upload: gradient", src=cid, dst="server")    # ... and record it
with timer("client: local update", who=cid):                     # wall-clock time of a step
    ...
```

`timer(...):` encapsulates an expression which is timed for execution.

`measure` sizes any nested object you are likely going to encounter:  

| object | size |
|---|---|
| torch tensor / NumPy array | `numel × element bits`, or `bits_per_entry`, e.g. 32 for values in `Z_{2^32}` |
| `bytes` (keys, ciphertexts, Shamir shares) | 8 bits per byte |
| `int` | minimal number of whole bytes |
| `float` | 64 bits |
| `str` | UTF-8 bytes |
| `dict` / `list` / `tuple`, dataclasses, `nn.Module` | sum of their parts |

`Run` activates a `Bench` per experiment that knows the current round. `report.txt` then logs everything in detail.

## Networks and datasets

**Datasets:** (`data.dataset`) are`mnist`, `fashion_mnist`, `cifar10`,  `digits` `synthetic`

You can partition the data with iid, dirichlet and just splitting it apart `data.shards_per_client`.Note `partition.png` in the outputs

**Networks**: 

- A simple regressor
- A multi level perceptron
- lenet
- A CNN
- A larger CNN
- Resnet18

## 12. Experiments

- Note that there are config yamls already present. They should be self-explanatory

The scripts are:

| script | configs | what it does |
|---|---|---|
| `fedavg.py` | `quick_digits`, `fedavg_mnist_iid `, `fedavg_mnist_dirichlet`, `fedsgd_mnist`, `fedavg_cifar10` | FedAvg vs FedSGD, IID vs non-IID |
| `byzantine.py` | `byz_ipm_mean`, `byz_alie_mnist`, `byz_fltrust_mnist`, `byz_labelflip_mnist`, `byz_backdoor_mnist`, byz_base_mnist` | attacks vs robust rules |
| `sweep.py` | `sweep_byzantine_mnist` | attacks × aggregators heatmap |
| `dp_fedavg.py` | `dp_central_mnist`, `dp_local_mnist` | utility vs ε, central vs local |
| `secure_aggregation.py` | `secagg_mnist`, `secaggplus_mnist`, `lightsecagg_mnist`, `lightsecagg_mnist_dropouts`, `secagg_dp_mnist` | training under SecAgg (+ dropouts, + distributed DP) |
| `secagg_scaling.py` | `secagg_scaling` | protocol cost vs. number of clients |

Then there is `privacy_attacks.py` with which you can play around to with model inversion attacks. Note that his is not fully developed

