"""Benchmarking: communication (bits) and computation (seconds) with two one-liners.

    from flcourse.bench import measure, timer

    measure(gradient, "upload/gradient", src=cid, dst="server")   # -> number of bits
    with timer("client/local_update", who=cid):
        ...

* :func:`measure` returns the estimated number of bits needed to serialise *any*
  (nested) Python object -- tensors, arrays, bytes (keys, ciphertexts, secret shares),
  ints, dicts, lists, ... -- and records it in the active :class:`Bench`.
* :func:`timer` is a context manager (or decorator) that records wall-clock time.

Both record into the *active* :class:`Bench` (``Run`` activates one per experiment)
together with the current round number. If no bench is active they still work (measure
returns the bits, timer just times) -- handy in notebooks and unit tests.

``who`` / ``src`` / ``dst`` are ``"server"`` or a client id (int). Everything that is not
``"server"`` counts as a client. :meth:`Bench.latency_estimate` uses this to estimate the
duration of a round *as if clients ran concurrently*: for every label it takes the
maximum over clients (instead of the sum) and converts bits into seconds with the
configured up/down link bandwidth.
"""
from __future__ import annotations

import dataclasses
import math
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import numpy as np
import torch

_ACTIVE: Optional["Bench"] = None


# ----------------------------------------------------------------------------------
# bit estimation
# ----------------------------------------------------------------------------------
def num_bits(obj: Any, bits_per_entry: Optional[int] = None) -> int:
    """Estimated size in bits of a serialised object (payload only, no framing).

    ``bits_per_entry`` overrides the per-element size of arrays/tensors, e.g.
    ``bits_per_entry=32`` for values that live in Z_{2^32} but are stored as int64.
    """
    if obj is None:
        return 0
    if isinstance(obj, bool):
        return 8
    if isinstance(obj, int):                       # minimal number of whole bytes (signed)
        return 8 * max(1, math.ceil((obj.bit_length() + 1) / 8))
    if isinstance(obj, float):
        return 64
    if isinstance(obj, complex):
        return 128
    if isinstance(obj, str):
        return 8 * len(obj.encode("utf-8"))
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return 8 * len(obj)
    if isinstance(obj, torch.Tensor):
        per = bits_per_entry if bits_per_entry is not None else obj.element_size() * 8
        return int(obj.numel()) * int(per)
    if isinstance(obj, np.ndarray):
        per = bits_per_entry if bits_per_entry is not None else obj.dtype.itemsize * 8
        return int(obj.size) * int(per)
    if isinstance(obj, np.generic):
        return obj.dtype.itemsize * 8
    if isinstance(obj, torch.nn.Module):
        return sum(num_bits(t, bits_per_entry) for t in obj.state_dict().values())
    if hasattr(obj, "__num_bits__"):
        return int(obj.__num_bits__())
    if isinstance(obj, dict):
        return sum(num_bits(k) + num_bits(v, bits_per_entry) for k, v in obj.items())
    if isinstance(obj, (list, tuple, set, frozenset)):
        return sum(num_bits(v, bits_per_entry) for v in obj)
    if dataclasses.is_dataclass(obj):
        return sum(num_bits(getattr(obj, f.name), bits_per_entry) for f in dataclasses.fields(obj))
    raise TypeError(f"measure(): don't know how to size an object of type {type(obj).__name__}; "
                    "convert it to bytes/arrays or give it a __num_bits__() method.")


def measure(obj: Any, label: str = "data", src: Any = None, dst: Any = None,
            bits_per_entry: Optional[int] = None) -> int:
    """Return the estimated bits to serialise ``obj`` and record them in the active Bench."""
    bits = num_bits(obj, bits_per_entry)
    if _ACTIVE is not None:
        _ACTIVE.add_comm(label, bits, src, dst)
    return bits


def _sync_device() -> None:
    """Make GPU timings meaningful (CUDA kernels run asynchronously)."""
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


@contextmanager
def timer(label: str, who: Any = "server"):
    """``with timer("server/aggregate"): ...`` -- record the elapsed wall-clock time."""
    _sync_device()
    start = time.perf_counter()
    try:
        yield
    finally:
        _sync_device()
        if _ACTIVE is not None:
            _ACTIVE.add_time(label, who, time.perf_counter() - start)


def active_bench() -> Optional["Bench"]:
    return _ACTIVE


# ----------------------------------------------------------------------------------
# formatting helpers
# ----------------------------------------------------------------------------------
def fmt_bits(bits: float) -> str:
    """Human readable size: bits are shown as bytes (B, kB, MB, GB; 1 kB = 1000 B)."""
    b = bits / 8.0
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if abs(b) < 1000 or unit == "TB":
            return f"{b:.0f} {unit}" if unit == "B" else f"{b:.2f} {unit}"
        b /= 1000.0
    return f"{b:.2f} TB"


def fmt_seconds(s: float) -> str:
    if s < 1e-3:
        return f"{s * 1e6:.0f} us"
    if s < 1:
        return f"{s * 1e3:.1f} ms"
    if s < 120:
        return f"{s:.2f} s"
    return f"{s / 60:.1f} min"


def kind(party: Any) -> str:
    """'server', 'client' or another named party (e.g. 'colluders')."""
    if party is None or party == "server":
        return "server"
    if isinstance(party, (int, np.integer)):
        return "client"
    if isinstance(party, str) and party.startswith("client"):
        return "client"
    return str(party)


def _table(rows: List[List[str]], header: List[str]) -> str:
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)] if rows else [len(h) for h in header]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    out = [line, "-" * len(line)]
    for row in rows:
        out.append("  ".join(str(x).ljust(w) if i == 0 else str(x).rjust(w)
                             for i, (x, w) in enumerate(zip(row, widths))))
    return "\n".join(out)


# ----------------------------------------------------------------------------------
# the recorder
# ----------------------------------------------------------------------------------
class Bench:
    """Collects communication and timing records of one experiment."""

    def __init__(self, name: str = ""):
        self.name = name
        self.round = 0
        self.comm: List[dict] = []
        self.time: List[dict] = []

    # -- lifecycle ---------------------------------------------------------------------
    def activate(self) -> "Bench":
        global _ACTIVE
        _ACTIVE = self
        return self

    def deactivate(self) -> None:
        global _ACTIVE
        if _ACTIVE is self:
            _ACTIVE = None

    def __enter__(self):
        return self.activate()

    def __exit__(self, *exc):
        self.deactivate()

    def set_round(self, rnd: int) -> None:
        self.round = int(rnd)

    # -- recording ---------------------------------------------------------------------
    def add_comm(self, label: str, bits: int, src: Any = None, dst: Any = None) -> None:
        self.comm.append({"round": self.round, "label": label, "bits": int(bits),
                          "src": _plain(src), "dst": _plain(dst)})

    def add_time(self, label: str, who: Any, seconds: float) -> None:
        self.time.append({"round": self.round, "label": label, "who": _plain(who), "seconds": float(seconds)})

    # -- summaries ---------------------------------------------------------------------
    @property
    def num_rounds(self) -> int:
        rounds = {r["round"] for r in self.comm} | {r["round"] for r in self.time}
        return max(len(rounds), 1)

    def comm_summary(self) -> List[dict]:
        """One row per (label, direction): messages, total bits, bits per message."""
        groups: Dict[tuple, List[dict]] = defaultdict(list)
        for r in self.comm:
            groups[(r["label"], f"{kind(r['src'])}->{kind(r['dst'])}")].append(r)
        rows = []
        for (label, direction), recs in groups.items():
            total = sum(r["bits"] for r in recs)
            rounds = {r["round"] for r in recs}
            rows.append({"label": label, "direction": direction, "messages": len(recs), "bits": total,
                         "bits_per_message": total / len(recs), "bits_per_round": total / max(len(rounds), 1)})
        return sorted(rows, key=lambda r: r["label"])

    def time_summary(self) -> List[dict]:
        """One row per label: total time, and per round the *average* and *max* over parties."""
        groups: Dict[str, List[dict]] = defaultdict(list)
        for r in self.time:
            groups[r["label"]].append(r)
        rows = []
        for label, recs in groups.items():
            per_round: Dict[int, Dict[Any, float]] = defaultdict(lambda: defaultdict(float))
            for r in recs:
                per_round[r["round"]][str(r["who"])] += r["seconds"]
            avgs = [float(np.mean(list(d.values()))) for d in per_round.values()]
            maxs = [max(d.values()) for d in per_round.values()]
            kinds = {kind(r["who"]) for r in recs}
            rows.append({"label": label, "who": "/".join(sorted(kinds)), "calls": len(recs),
                         "total_s": sum(r["seconds"] for r in recs), "mean_call_s": np.mean([r["seconds"] for r in recs]),
                         "avg_per_round_s": float(np.mean(avgs)), "parallel_per_round_s": float(np.mean(maxs))})
        return sorted(rows, key=lambda r: r["label"])

    def latency_estimate(self, uplink_mbps: float = 10.0, downlink_mbps: float = 50.0) -> dict:
        """Estimated duration of a round if all clients ran concurrently.

        Every label is treated as one protocol phase; within a phase clients work in
        parallel (max over clients), phases are sequential (sum over labels).
        Communication time of a client = bits sent / uplink + bits received / downlink.
        The server is assumed to have unlimited bandwidth.
        """
        up, down = uplink_mbps * 1e6, downlink_mbps * 1e6
        per_round: Dict[int, Dict[str, float]] = defaultdict(lambda: {"compute": 0.0, "comm": 0.0})

        by_round_label: Dict[tuple, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for r in self.time:
            by_round_label[(r["round"], r["label"])][str(r["who"])] += r["seconds"]
        for (rnd, _), parties in by_round_label.items():
            per_round[rnd]["compute"] += max(parties.values())

        comm_by: Dict[tuple, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for r in self.comm:
            key = (r["round"], r["label"])
            if kind(r["src"]) == "client":
                comm_by[key][f"c{r['src']}"] += r["bits"] / up
            if kind(r["dst"]) == "client":
                comm_by[key][f"c{r['dst']}"] += r["bits"] / down
        for (rnd, _), parties in comm_by.items():
            per_round[rnd]["comm"] += max(parties.values())

        if not per_round:
            return {"compute_s": 0.0, "comm_s": 0.0, "total_s": 0.0, "rounds": 0}
        comp = float(np.mean([v["compute"] for v in per_round.values()]))
        comm = float(np.mean([v["comm"] for v in per_round.values()]))
        return {"compute_s": comp, "comm_s": comm, "total_s": comp + comm, "rounds": len(per_round),
                "uplink_mbps": uplink_mbps, "downlink_mbps": downlink_mbps}

    def per_round_comm(self) -> Dict[int, int]:
        out: Dict[int, int] = defaultdict(int)
        for r in self.comm:
            out[r["round"]] += r["bits"]
        return dict(sorted(out.items()))

    # -- text report ---------------------------------------------------------------------
    def report(self, uplink_mbps: float = 10.0, downlink_mbps: float = 50.0) -> str:
        lines = []
        comm_rows = self.comm_summary()
        if comm_rows:
            total = sum(r["bits"] for r in comm_rows)
            lines.append(f"COMMUNICATION   (total {fmt_bits(total)} over {self.num_rounds} round(s))")
            rows = [[r["label"], r["direction"], r["messages"], fmt_bits(r["bits"]),
                     fmt_bits(r["bits_per_round"]), fmt_bits(r["bits_per_message"])] for r in comm_rows]
            lines.append(_table(rows, ["label", "direction", "msgs", "total", "per round", "per msg"]))
        time_rows = self.time_summary()
        if time_rows:
            lines.append("")
            lines.append("COMPUTATION   ('avg/round' = average over parties, "
                         "'parallel/round' = max over parties = concurrent clients)")
            rows = [[r["label"], r["who"], r["calls"], fmt_seconds(r["total_s"]), fmt_seconds(r["mean_call_s"]),
                     fmt_seconds(r["avg_per_round_s"]), fmt_seconds(r["parallel_per_round_s"])] for r in time_rows]
            lines.append(_table(rows, ["label", "who", "calls", "total", "per call", "avg/round", "parallel/round"]))
        if comm_rows or time_rows:
            est = self.latency_estimate(uplink_mbps, downlink_mbps)
            lines.append("")
            lines.append(f"ESTIMATED ROUND LATENCY with concurrent clients (uplink {uplink_mbps:g} Mbit/s, "
                         f"downlink {downlink_mbps:g} Mbit/s): compute {fmt_seconds(est['compute_s'])} + "
                         f"communication {fmt_seconds(est['comm_s'])} = {fmt_seconds(est['total_s'])} per round")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"comm": self.comm, "time": self.time}


def _plain(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    return x
