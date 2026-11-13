"""flcourse -- a small, readable federated learning framework for teaching.

Main entry points

* ``flcourse.aggregators`` / ``flcourse.preaggregators``  robust aggregation (ByzFL style)
* ``flcourse.attacks``     model/data poisoning, collusion channel, gradient inversion
* ``flcourse.dp``          clipping, Gaussian noise, Rényi-DP accountant
* ``flcourse.secagg``      SecAgg, SecAgg+ (Flower primitives), LightSecAgg
* ``flcourse.bench``       ``measure(obj)`` (bits) and ``timer(label)`` (seconds)
* ``flcourse.run``         ``Run`` (output folder, metrics, report) and ``build_federation``

The training loops themselves live in ``experiments/`` -- one readable file per experiment.
"""
from .aggregators import AGGREGATORS, make_aggregator
from .backend import as_matrix, stack, to_backend, to_numpy, to_torch
from .bench import Bench, measure, num_bits, timer
from .config import Config, load_config, parse_args
from .device import get_device, set_seed
from .preaggregators import PRE_AGGREGATORS, make_pre_aggregators

__version__ = "0.1.0"

__all__ = ["make_aggregator", "make_pre_aggregators", "AGGREGATORS", "PRE_AGGREGATORS", "measure", "timer",
           "num_bits", "Bench", "load_config", "parse_args", "Config", "get_device", "set_seed", "stack", "as_matrix",
           "to_numpy", "to_torch", "to_backend"]
