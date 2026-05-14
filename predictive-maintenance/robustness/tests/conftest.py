"""
Shared fixtures for the robustness module test suite.

All arrays use the NASA C-MAPSS format: 21 sensor channels, 30 time cycles,
float64 dtype.  The ``rng`` fixture pins the seed so every test that derives
data from it is deterministic.
"""

import sys
import os

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Make `robustness` importable when running pytest from the repo root or from
# this directory directly.
# ---------------------------------------------------------------------------
_PREDICTIVE_MAINTENANCE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PREDICTIVE_MAINTENANCE_DIR not in sys.path:
    sys.path.insert(0, _PREDICTIVE_MAINTENANCE_DIR)

# ---------------------------------------------------------------------------
# C-MAPSS constants
# ---------------------------------------------------------------------------
N_SENSORS: int = 21
N_CYCLES: int = 30


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Session-scoped RNG with a fixed seed — fully reproducible."""
    return np.random.default_rng(42)


@pytest.fixture
def cmapss_batch(rng) -> np.ndarray:
    """
    30-cycle × 21-sensor array matching normalised C-MAPSS format.

    Values are drawn from N(0, 1) — analogous to z-score normalised sensor
    data.  Shape: ``(30, 21)``, dtype: ``float64``.
    """
    return rng.standard_normal((N_CYCLES, N_SENSORS))


@pytest.fixture
def cmapss_single(rng) -> np.ndarray:
    """
    Single 21-sensor reading (1-D, shape ``(21,)``).

    Used to test single-timestep code paths in all three modules.
    """
    return rng.standard_normal(N_SENSORS)
