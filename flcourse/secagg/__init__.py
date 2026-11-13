"""Secure aggregation: SecAgg, SecAgg+ (on Flower's primitives) and LightSecAgg.

All protocols share one interface::

    secagg = make_secure_aggregator({"protocol": "secagg+", "num_shares": 5, "reconstruction_threshold": 3})
    mean = secagg.aggregate({client_id: vector, ...}, dropouts={client_id: "masked_input"})

``mean`` is the (unweighted, or weighted for SecAgg/SecAgg+) average of the inputs of
the clients that did not drop out; the server never sees an individual input.
"""
from .lightsecagg import LightSecAgg, LightSecAggParams, lightsecagg_aggregate
from .secaggplus import SecAgg, SecAggPlus, SecAggPlusParams, secaggplus_aggregate

PROTOCOLS = {"secagg": SecAgg, "secagg+": SecAggPlus, "secaggplus": SecAggPlus, "lightsecagg": LightSecAgg}


def make_secure_aggregator(spec, seed=None):
    """Build a protocol from a config section, e.g. ``{"protocol": "lightsecagg", "privacy_threshold": 0.3}``.
    Keys other than ``protocol``, ``dropout_rate``, ``dropout_stage`` and ``recovery_dropout_rate``
    (dropout simulation, see experiments/secure_aggregation.py) are protocol parameters."""
    if spec is None:
        return None
    spec = dict(spec)
    name = str(spec.pop("protocol", "secagg")).lower()
    for key in ("dropout_rate", "dropout_stage", "recovery_dropout_rate", "enabled"):
        spec.pop(key, None)
    if name not in PROTOCOLS:
        raise ValueError(f"Unknown secure aggregation protocol {name!r}. Available: {sorted(PROTOCOLS)}")
    if name != "lightsecagg":
        spec.setdefault("seed", seed)
    return PROTOCOLS[name](**spec)


__all__ = ["SecAgg", "SecAggPlus", "SecAggPlusParams", "secaggplus_aggregate", "LightSecAgg", "LightSecAggParams",
           "lightsecagg_aggregate", "make_secure_aggregator", "PROTOCOLS"]
