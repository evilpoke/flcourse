import numpy as np
import pytest
import torch

from flcourse.backend import (NUMPY, TORCH, as_matrix, pairwise_sq_distances, same_type_as, stack, to_backend,
                              to_numpy, to_torch)


def test_as_matrix_accepts_all_input_kinds():
    rows = [np.arange(3.0), np.ones(3)]
    for vectors, expected_ops in [(rows, NUMPY), (np.stack(rows), NUMPY),
                                  ([torch.tensor(r) for r in rows], TORCH), (torch.tensor(np.stack(rows)), TORCH)]:
        X, xp = as_matrix(vectors)
        assert xp is expected_ops
        assert tuple(X.shape) == (2, 3)


def test_as_matrix_rejects_wrong_shapes_and_empty_lists():
    with pytest.raises(ValueError):
        as_matrix(np.zeros(3))
    with pytest.raises(ValueError):
        as_matrix([])


def test_median_and_std_identical_for_numpy_and_torch(rng):
    X = rng.standard_normal((6, 5))          # even number of rows: torch.median would differ
    np.testing.assert_allclose(NUMPY.median(X), np.median(X, axis=0))
    np.testing.assert_allclose(TORCH.median(torch.tensor(X)).numpy(), np.median(X, axis=0))
    np.testing.assert_allclose(TORCH.std(torch.tensor(X)).numpy(), np.std(X, axis=0, ddof=1))
    np.testing.assert_allclose(NUMPY.std(X), np.std(X, axis=0, ddof=1))


def test_pairwise_distances(rng):
    X = rng.standard_normal((5, 7))
    D = pairwise_sq_distances(X)
    expected = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    np.testing.assert_allclose(D, expected, atol=1e-12)
    np.testing.assert_allclose(pairwise_sq_distances(torch.tensor(X)).numpy(), expected, atol=1e-12)


def test_conversions_roundtrip(rng):
    x = rng.standard_normal(4).astype(np.float32)
    t = to_torch(x)
    assert isinstance(t, torch.Tensor) and t.dtype == torch.float32
    np.testing.assert_array_equal(to_numpy(t), x)
    assert isinstance(to_backend(t, "numpy"), np.ndarray)
    assert isinstance(to_backend(x, "torch"), torch.Tensor)
    assert isinstance(same_type_as(x, t), torch.Tensor)
    assert isinstance(stack([t, t]), torch.Tensor)
    with pytest.raises(ValueError):
        to_backend(x, "jax")
