import numpy as np
import pytest

from flcourse.bench import Bench
from flcourse.secagg import LightSecAgg, SecAgg, SecAggPlus, SecAggPlusParams, make_secure_aggregator
from flcourse.secagg.flwr.quantization import dequantize, quantize
from flcourse.secagg.flwr.shamir import combine_shares, create_shares
from flcourse.secagg.lightsecagg import PRIME, field, t_private_mds_matrix
from flcourse.secagg.secaggplus import build_neighbours, round_parameters

N, D = 8, 300
STEP = 2 * 1.0 / 2**22          # quantisation step for clipping_range = 1


@pytest.fixture
def inputs(rng):
    return {i: rng.uniform(-1, 1, D) for i in range(N)}


def mean_of(inputs, ids):
    return np.mean([inputs[i] for i in ids], axis=0)


# ---- vendored Flower primitives -------------------------------------------------------------
def test_shamir_threshold():
    secret = bytes(range(40))
    shares = create_shares(secret, threshold=3, num=5)
    assert combine_shares(shares[:3]) == secret
    assert combine_shares([shares[4], shares[1], shares[2]]) == secret
    try:
        wrong = combine_shares(shares[:2])
    except ValueError:
        wrong = None
    assert wrong != secret


def test_quantization_roundtrip(rng):
    x = rng.uniform(-2, 2, 1000)
    q = quantize([x], clipping_range=2.0, target_range=2**16)[0]
    assert q.min() >= 0 and q.max() <= 2**16
    back = dequantize([q], 2.0, 2**16)[0]
    assert np.max(np.abs(back - x)) <= 4.0 / 2**16 + 1e-12


# ---- SecAgg / SecAgg+ ----------------------------------------------------------------------
@pytest.mark.parametrize("proto", [SecAgg(0.5, clipping_range=1.0, seed=0),
                                   SecAggPlus(num_shares=5, reconstruction_threshold=3, clipping_range=1.0, seed=0)],
                         ids=["secagg", "secagg+"])
def test_secure_mean_with_and_without_dropouts(proto, inputs):
    out = proto.aggregate(inputs)
    assert np.max(np.abs(out - mean_of(inputs, range(N)))) < 2 * STEP
    for dropouts in ({2: "masked_input"}, {2: "masked_input", 5: "unmask"}, {1: "share_keys", 6: "masked_input"}):
        out = proto.aggregate(inputs, dropouts=dropouts)
        survivors = [i for i in range(N) if dropouts.get(i) not in ("masked_input", "share_keys")]
        assert proto.last_info["aggregated_clients"] == survivors
        assert np.max(np.abs(out - mean_of(inputs, survivors))) < 2 * STEP


def test_secagg_weighted_mean(inputs):
    w = {i: float(i + 1) for i in range(N)}
    out = SecAgg(0.5, clipping_range=1.0, max_weight=float(N), seed=0).aggregate(inputs, weights=w)
    expected = sum(w[i] * inputs[i] for i in range(N)) / sum(w.values())
    assert np.max(np.abs(out - expected)) < 1e-4


def test_secagg_fails_with_too_many_dropouts(inputs):
    with pytest.raises(RuntimeError):
        SecAgg(0.5, seed=0).aggregate(inputs, dropouts={i: "unmask" for i in range(5)})


def test_flower_parameter_checks():
    with pytest.raises(ValueError):
        SecAggPlusParams(num_shares=2)
    with pytest.raises(ValueError):
        SecAggPlusParams(num_shares=5, reconstruction_threshold=5)
    with pytest.raises(ValueError):
        SecAggPlusParams(modulus_range=1000)                       # not a power of 2
    with pytest.raises(ValueError):
        SecAggPlusParams(quantization_range=2**33, modulus_range=2**32)
    with pytest.raises(ValueError):                                # n * T >= M would overflow Z_M
        round_parameters(SecAggPlusParams(quantization_range=2**30, modulus_range=2**32), 8)
    assert round_parameters(SecAggPlusParams(num_shares=1.0, reconstruction_threshold=0.5), 10) == (10, 5)
    assert round_parameters(SecAggPlusParams(num_shares=0.5, reconstruction_threshold=0.5), 10) == (5, 2)


def test_secaggplus_graph_is_symmetric():
    import random
    nb = build_neighbours(range(11), 5, random.Random(0))
    assert all(len(v) == 5 and i in v for i, v in nb.items())
    assert all(i in nb[j] for i in nb for j in nb[i])


def test_secagg_communication_is_measured(inputs):
    bench = Bench().activate()
    try:
        SecAgg(0.5, seed=0).aggregate(inputs)
    finally:
        bench.deactivate()
    labels = {r["label"] for r in bench.comm}
    assert {"secagg/0 setup: public keys", "secagg/1 share keys: encrypted shares",
            "secagg/2 masked input: masked vector", "secagg/3 unmask: secret shares"} <= labels
    masked = [r for r in bench.comm if r["label"] == "secagg/2 masked input: masked vector"]
    assert masked[0]["bits"] == (D + 1) * 32                      # d entries + weight factor, 32 bit each


# ---- LightSecAgg ---------------------------------------------------------------------------
def test_encoding_matrix_is_t_private_mds():
    """Definition of the paper (Sec. 4.1, footnote 2): W (U x N) is MDS if every U x U submatrix
    is non-singular, and T-private if additionally its last T rows form an MDS matrix."""
    from itertools import combinations
    for n in range(2, 8):
        for u in range(1, n + 1):
            W = t_private_mds_matrix(u, n)
            assert W.shape == (u, n)
            assert all(np.linalg.matrix_rank(W[:, list(c)]) == u for c in combinations(range(n), u))
            for t in range(1, u):
                assert all(np.linalg.matrix_rank(W[u - t:, list(c)]) == t for c in combinations(range(n), t))


def test_mds_encode_decode_roundtrip(rng):
    GF = field(PRIME)
    U, N = 4, 7
    W = t_private_mds_matrix(U, N)
    pieces = GF(rng.integers(0, PRIME, size=(U, 9)))
    coded = W.T @ pieces                                   # row j = pieces . W_j  (eq. 5 of the paper)
    cols = [1, 3, 4, 6]                                    # any U of the N coded rows suffice
    back = np.linalg.inv(W[:, cols].T) @ coded[cols]
    np.testing.assert_array_equal(back, pieces)


def test_lightsecagg(inputs):
    proto = LightSecAgg(privacy_threshold=2, dropout_tolerance=2, clipping_range=1.0)
    assert np.max(np.abs(proto.aggregate(inputs) - mean_of(inputs, range(N)))) < 2 * STEP
    out = proto.aggregate(inputs, dropouts={0: "masked_input", 3: "recovery"})
    assert np.max(np.abs(out - mean_of(inputs, [i for i in range(N) if i != 0]))) < 2 * STEP
    with pytest.raises(RuntimeError):
        proto.aggregate(inputs, dropouts={0: "masked_input", 1: "masked_input", 2: "masked_input"})
    with pytest.raises(ValueError):
        LightSecAgg(privacy_threshold=6, dropout_tolerance=2).aggregate(inputs)    # needs T < U


def test_factory():
    assert isinstance(make_secure_aggregator({"protocol": "secagg+", "num_shares": 5,
                                              "reconstruction_threshold": 3, "dropout_rate": 0.1}), SecAggPlus)
    assert isinstance(make_secure_aggregator({"protocol": "lightsecagg"}), LightSecAgg)
    assert make_secure_aggregator(None) is None
    with pytest.raises(ValueError):
        make_secure_aggregator({"protocol": "magic"})

