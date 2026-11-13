import numpy as np
import pytest
import torch

from flcourse.aggregators import (AGGREGATORS, Average, Bulyan, CenteredClipping, FLTrust, GeometricMedian, Krum,
                                  Median, MultiKrum, TrimmedMean, WeightedAverage, make_aggregator)
from flcourse.preaggregators import ARC, NNM, Bucketing, Clipping, make_pre_aggregators

STATELESS = [Average(), Median(), TrimmedMean(f=4), GeometricMedian(), Krum(f=4), MultiKrum(f=4), Bulyan(f=4)]


@pytest.mark.parametrize("agg", STATELESS, ids=lambda a: type(a).__name__)
def test_numpy_and_torch_give_the_same_result(agg, honest_and_outliers):
    _, X = honest_and_outliers
    a = agg(X)
    b = agg(torch.tensor(X))
    c = agg([torch.tensor(x) for x in X])
    assert isinstance(a, np.ndarray) and isinstance(b, torch.Tensor)
    np.testing.assert_allclose(a, b.numpy(), atol=1e-8)
    np.testing.assert_allclose(a, c.numpy(), atol=1e-8)


@pytest.mark.parametrize("agg", STATELESS[1:], ids=lambda a: type(a).__name__)
def test_robust_rules_ignore_far_outliers(agg, honest_and_outliers):
    honest, X = honest_and_outliers
    out = agg(X)
    assert np.linalg.norm(out - honest.mean(0)) < 0.5            # stays with the honest majority
    assert np.linalg.norm(Average()(X) - honest.mean(0)) > 10    # the mean is dragged away


def test_simple_known_values():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [100.0, -100.0]])
    np.testing.assert_allclose(Average()(X), [26.5, -10.0])
    np.testing.assert_allclose(Median()(X), [2.5, 15.0])
    np.testing.assert_allclose(TrimmedMean(f=1)(X), [2.5, 15.0])
    np.testing.assert_allclose(WeightedAverage()(X[:3], weights=[1, 1, 2]), [2.25, 22.5])


def test_krum_returns_one_of_the_inputs(honest_and_outliers):
    _, X = honest_and_outliers
    out = Krum(f=4)(X)
    assert any(np.array_equal(out, x) for x in X)


def test_size_requirements_are_checked():
    X = np.zeros((6, 3))
    with pytest.raises(ValueError):
        Krum(f=2)(X)         # needs n >= 2f + 3 = 7
    with pytest.raises(ValueError):
        Bulyan(f=1)(X)       # needs n >= 4f + 3 = 7
    with pytest.raises(ValueError):
        TrimmedMean(f=3)(X)  # needs n >= 2f + 1 = 7


def test_centered_clipping_moves_at_most_tau_per_iteration():
    X = np.full((5, 4), 100.0)
    agg = CenteredClipping(tau=1.0, L=1)
    out = agg(X)
    assert np.linalg.norm(out) == pytest.approx(1.0)
    out2 = agg(X)                                   # stateful: starts from the previous output
    assert np.linalg.norm(out2) == pytest.approx(2.0)
    agg.reset()
    assert np.linalg.norm(agg(torch.tensor(X)).numpy()) == pytest.approx(1.0)


def test_fltrust_distrusts_opposite_directions():
    ref = np.array([1.0, 0.0])
    X = np.array([[2.0, 0.0], [4.0, 0.1], [-10.0, 0.0]])
    out = FLTrust()(X, reference=ref)
    assert out[0] > 0.9                    # rescaled to |ref| and averaged over trusted vectors
    assert np.linalg.norm(out) <= np.linalg.norm(ref) + 1e-9
    np.testing.assert_allclose(FLTrust()(torch.tensor(X), reference=torch.tensor(ref)).numpy(), out)
    assert np.allclose(FLTrust()(-X[:2], reference=ref), 0.0)     # nobody trusted -> zero update
    with pytest.raises(ValueError):
        FLTrust()(X)


def test_registry_and_f_default():
    agg = make_aggregator({"name": "krum"}, f_default=3)
    assert isinstance(agg, Krum) and agg.f == 3
    assert make_aggregator({"name": "trimmed_mean", "f": 1}, f_default=3).f == 1
    assert isinstance(make_aggregator(None), Average)
    with pytest.raises(ValueError):
        make_aggregator({"name": "does_not_exist"})
    with pytest.raises(ValueError):
        make_aggregator({"name": "median", "tau": 1.0})       # unknown parameter
    assert {"mean", "median", "krum", "bulyan", "fltrust", "centered_clipping"} <= set(AGGREGATORS)


def test_preaggregators(honest_and_outliers, rng):
    _, X = honest_and_outliers
    for pre in [NNM(f=4), Clipping(c=1.0), ARC(f=4)]:
        a, b = pre(X), pre(torch.tensor(X))
        assert a.shape == X.shape
        np.testing.assert_allclose(a, b.numpy(), atol=1e-8)
    assert np.all(np.linalg.norm(Clipping(c=1.0)(X), axis=1) <= 1.0 + 1e-9)
    norms = np.linalg.norm(ARC(f=4)(X), axis=1)
    assert norms.max() <= np.sort(np.linalg.norm(X, axis=1))[-1]
    assert Bucketing(s=4)(X).shape == (5, 20)                 # ceil(19 / 4) buckets
    assert Bucketing(s=4)(torch.tensor(X)).shape == (5, 20)
    X_before = X.copy()
    NNM(f=4)(X), Clipping(c=1.0)(X), ARC(f=4)(X)
    np.testing.assert_array_equal(X, X_before)                # inputs are never modified
    pres = make_pre_aggregators([{"name": "nnm"}, {"name": "clipping", "c": 2.0}], f_default=2)
    assert isinstance(pres[0], NNM) and pres[0].f == 2 and pres[1].c == 2.0


def test_nnm_then_trimmed_mean_pipeline(honest_and_outliers):
    honest, X = honest_and_outliers
    Y = NNM(f=4)(X)
    out = TrimmedMean(f=4)(Y)
    assert np.linalg.norm(out - honest.mean(0)) < 0.5
