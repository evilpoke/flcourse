import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def honest_and_outliers(rng):
    """15 honest vectors around 1.0 and 4 far-away outliers at -50."""
    honest = 1.0 + 0.1 * rng.standard_normal((15, 20))
    outliers = np.full((4, 20), -50.0)
    return honest, np.concatenate([honest, outliers])


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)
    torch.manual_seed(0)
