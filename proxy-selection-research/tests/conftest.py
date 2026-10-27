import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def synthetic_two_model_scores():
    """Synthetic arrays only (never research data)."""
    rng = np.random.default_rng(20261026)
    y = np.array([0] * 20 + [1] * 20)
    s_a = np.clip(rng.normal(0.35, 0.15, 40) + 0.25 * y, 0, 1)
    s_b = np.clip(rng.normal(0.45, 0.15, 40) + 0.15 * y, 0, 1)
    groups = np.array([f"grp_{i:02d}" for i in range(40)])
    return {"y": y, "s_a": s_a, "s_b": s_b, "groups": groups}


@pytest.fixture(scope="session")
def synthetic_digital_matrix():
    return {
        "A": np.array([0.90, 0.60]),
        "B": np.array([0.60, 0.90]),
        "C": np.array([0.70, 0.70]),
    }


@pytest.fixture(scope="session")
def synthetic_dominant_matrix():
    return {
        "A": np.array([0.90, 0.86, 0.82]),
        "B": np.array([0.85, 0.80, 0.75]),
    }
