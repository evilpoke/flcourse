"""A shared channel for colluding parties.

Colluding clients (and possibly a curious server) need a way to exchange information
that is *not* part of the FL protocol. In the sequential simulation this is simply a
"blackboard" object that is created once per round and handed to every colluder in a
separate loop after the normal client loop::

    channel = CollusionChannel()
    for cid in colluders:                          # 1) everybody publishes what it knows
        channel.post(cid, uploads[cid])
    byzantine = attack(channel.values())           # 2) craft the malicious upload jointly
    for cid in colluders:
        uploads[cid] = byzantine

Everything posted is recorded by :func:`flcourse.bench.measure` (direction
``client->colluders``) so the cost of collusion shows up in the benchmark report.
Topics separate different kinds of information, e.g. ``"vector"`` for model updates or
``"secagg_shares"`` for secret shares handed to a curious server.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..bench import measure


class CollusionChannel:
    def __init__(self, name: str = "collusion"):
        self.name = name
        self.board: Dict[str, Dict[Any, Any]] = {}

    def post(self, sender: Any, value: Any, topic: str = "vector", record: bool = True) -> None:
        """Publish ``value`` under ``topic`` (``record=False``: do not count as communication)."""
        if record:
            measure(value, f"{self.name}/{topic}", src=sender, dst="colluders")
        self.board.setdefault(topic, {})[sender] = value

    def read(self, topic: str = "vector") -> Dict[Any, Any]:
        """``{sender: value}`` for one topic."""
        return dict(self.board.get(topic, {}))

    def values(self, topic: str = "vector") -> List[Any]:
        return list(self.board.get(topic, {}).values())

    def senders(self, topic: str = "vector") -> List[Any]:
        return list(self.board.get(topic, {}).keys())

    def __len__(self) -> int:
        return sum(len(v) for v in self.board.values())

    def __repr__(self) -> str:
        topics = {t: len(v) for t, v in self.board.items()}
        return f"CollusionChannel({self.name!r}, topics={topics})"
