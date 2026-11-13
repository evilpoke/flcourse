import time

import numpy as np
import pytest
import torch

from flcourse.bench import Bench, fmt_bits, measure, num_bits, timer
from flcourse.report import plot_communication, plot_computation


def test_num_bits_of_common_objects():
    assert num_bits(np.zeros(10, dtype=np.float32)) == 320
    assert num_bits(np.zeros(10, dtype=np.int64), bits_per_entry=32) == 320
    assert num_bits(torch.zeros(5, dtype=torch.float16)) == 80
    assert num_bits(b"abcd") == 32
    assert num_bits("abc") == 24
    assert num_bits(255) == 16 and num_bits(127) == 8          # signed, whole bytes
    assert num_bits(1.5) == 64 and num_bits(True) == 8 and num_bits(None) == 0
    assert num_bits({"key": b"ab", 7: [np.zeros(2, dtype=np.uint8), 1.0]}) == 24 + 16 + 8 + 16 + 64
    assert num_bits(torch.nn.Linear(3, 2)) == (6 + 2) * 32
    with pytest.raises(TypeError):
        num_bits(object())


def test_measure_and_timer_record_into_the_active_bench():
    assert measure(np.zeros(4, dtype=np.float32)) == 128          # no active bench: just returns bits
    bench = Bench()
    with bench:
        bench.set_round(1)
        measure(np.zeros(4, dtype=np.float32), "upload", src=0, dst="server")
        measure(np.zeros(8, dtype=np.float32), "upload", src=1, dst="server")
        with timer("client: work", who=0):
            time.sleep(0.01)
        with timer("client: work", who=1):
            time.sleep(0.03)
        with timer("server: aggregate"):
            pass
    measure(np.zeros(4), "ignored")                                 # bench is no longer active
    assert len(bench.comm) == 2 and len(bench.time) == 3
    (row,) = bench.comm_summary()
    assert row["bits"] == 384 and row["messages"] == 2 and row["direction"] == "client->server"
    t = {r["label"]: r for r in bench.time_summary()}["client: work"]
    assert t["avg_per_round_s"] == pytest.approx(0.02, abs=0.01)       # mean over the two clients
    assert t["parallel_per_round_s"] == pytest.approx(0.03, abs=0.015)     # max over the two clients
    est = bench.latency_estimate(uplink_mbps=1e-3, downlink_mbps=1.0)       # 1 kbit/s uplink
    assert est["comm_s"] == pytest.approx(256 / 1e3)                        # slowest client sends 256 bits
    text = bench.report()
    assert "COMMUNICATION" in text and "client: work" in text and "ESTIMATED ROUND LATENCY" in text


def test_plots(tmp_path):
    bench = Bench()
    with bench:
        for rnd in (1, 2):
            bench.set_round(rnd)
            measure(np.zeros(100, dtype=np.float32), "upload", src=0, dst="server")
            measure(np.zeros(100, dtype=np.float32), "download", src="server", dst=0)
            with timer("client: work", who=0):
                pass
    assert plot_communication(bench, tmp_path / "c.png").exists()
    assert plot_computation(bench, tmp_path / "t.png").exists()
    assert plot_communication(Bench(), tmp_path / "none.png") is None


def test_fmt_bits():
    assert fmt_bits(8) == "1 B" and fmt_bits(8 * 1500) == "1.50 kB" and fmt_bits(8 * 2.5e6) == "2.50 MB"
