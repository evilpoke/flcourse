import math

import numpy as np
import pytest
import torch

from flcourse.dp import (RDPAccountant, add_gaussian_noise, clip_l2, effective_noise_multiplier, epsilon_after,
                         gaussian_sigma_for_epsilon, l2_norm, noise_multiplier_for, privatize_mean,
                         rdp_subsampled_gaussian)

OPACUS_REFERENCE = [  # (sigma, q, steps, delta, epsilon)
    (1.0, 0.01, 1000, 1e-5, 2.1014),
    (1.1, 0.1, 100, 1e-5, 6.6137),
    (0.8, 0.3, 50, 1e-5, 24.5162),
    (2.0, 1.0, 10, 1e-6, 8.8469),
    (1.5, 0.05, 200, 1e-5, 2.6026),
]


def test_clip_l2():
    x = np.array([3.0, 4.0])
    np.testing.assert_allclose(clip_l2(x, 1.0), [0.6, 0.8])
    assert clip_l2(x, 10.0) is x                                   # short vectors are untouched
    t = clip_l2(torch.tensor([3.0, 4.0]), 2.5)
    assert isinstance(t, torch.Tensor) and l2_norm(t) == pytest.approx(2.5)


def test_gaussian_noise_statistics():
    x = np.zeros(200_000)
    noisy = add_gaussian_noise(x, std=0.5, generator=np.random.default_rng(0))
    assert noisy.std() == pytest.approx(0.5, rel=0.01) and abs(noisy.mean()) < 0.01
    t = add_gaussian_noise(torch.zeros(200_000), std=2.0, generator=torch.Generator().manual_seed(0))
    assert float(t.std()) == pytest.approx(2.0, rel=0.01)
    assert add_gaussian_noise(x, 0.0) is x


def test_rdp_closed_form_without_subsampling():
    orders = [2, 5, 32]
    np.testing.assert_allclose(rdp_subsampled_gaussian(1.0, 2.0, orders), [a / 8 for a in orders])
    assert np.all(rdp_subsampled_gaussian(0.0, 1.0, orders) == 0)


@pytest.mark.parametrize("sigma,q,steps,delta,opacus_eps", OPACUS_REFERENCE)
def test_accountant_matches_opacus(sigma, q, steps, delta, opacus_eps):
    acc = RDPAccountant()
    acc.step(sigma, q, steps)
    eps = acc.get_epsilon(delta)
    # integer orders only -> never smaller (i.e. never optimistic), at most 3% larger
    assert opacus_eps - 1e-3 <= eps <= 1.03 * opacus_eps


def test_accountant_monotonicity_and_inverse():
    assert epsilon_after(1.0, 0.1, 200, 1e-5) > epsilon_after(1.0, 0.1, 100, 1e-5)
    assert epsilon_after(2.0, 0.1, 100, 1e-5) < epsilon_after(1.0, 0.1, 100, 1e-5)
    assert epsilon_after(1.0, 0.05, 100, 1e-5) < epsilon_after(1.0, 0.1, 100, 1e-5)
    sigma = noise_multiplier_for(3.0, 1e-5, 0.1, 100)
    assert epsilon_after(sigma, 0.1, 100, 1e-5) <= 3.0 < epsilon_after(sigma - 0.05, 0.1, 100, 1e-5)
    acc = RDPAccountant()
    for _ in range(10):
        acc.step(1.0, 0.1)
    assert acc.get_epsilon(1e-5) == pytest.approx(epsilon_after(1.0, 0.1, 10, 1e-5))


def test_helpers():
    assert gaussian_sigma_for_epsilon(0.5, 1e-5) == pytest.approx(math.sqrt(2 * math.log(1.25e5)) / 0.5)
    assert effective_noise_multiplier(1.0, 10, 0) == 1.0
    assert effective_noise_multiplier(1.0, 10, 9) == pytest.approx(math.sqrt(0.1))
    X = np.ones((4, 3)) * 10
    out = privatize_mean(X, max_norm=1.0, noise_multiplier=0.0)
    assert l2_norm(out) == pytest.approx(1.0)
