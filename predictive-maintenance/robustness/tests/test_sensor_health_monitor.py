"""
Unit tests for SensorHealthMonitor.

All tests that depend on TensorFlow are skipped when TF is not installed via
``pytest.importorskip``.  The autoencoder's ``predict`` method is replaced in
fixtures with a deterministic lambda so tests never perform actual gradient
steps and execute in milliseconds.

Coverage:
- Health scores are always in [0.0, 1.0].
- Perfect reconstruction (zero error) → score 1.0.
- Maximum reconstruction error ≥ threshold → score 0.0.
- Per-sensor independence: different errors produce different scores.
- Temporal aggregation: 2-D input (window) works correctly.
- fit() populates _calibration_thresholds from training error percentiles.
- compute_health_scores() raises RuntimeError before fit().
- identify_degraded_sensors() returns the correct indices.
- save() writes a calibration_thresholds.npy file.
- load() restores thresholds from disk; raises FileNotFoundError if missing.
- Input validation: dtype, shape, NaN/Inf.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Skip the entire module when TensorFlow is unavailable.
# We catch ValueError too because a numpy/h5py ABI mismatch raises ValueError
# rather than ImportError when TF tries to load h5py.
try:
    import tensorflow as _tf  # noqa: F401
except (ImportError, ValueError) as _tf_err:
    pytest.skip(
        f"TensorFlow not available ({_tf_err}); skipping SensorHealthMonitor tests.",
        allow_module_level=True,
    )

from robustness.sensor_health_monitor import SensorHealthMonitor

N_SENSORS = 21
N_CYCLES = 30


# ===========================================================================
# Fixtures
# ===========================================================================


def _make_mock_ae(reconstruct_fn=None) -> MagicMock:
    """
    Return a mock autoencoder.

    Parameters
    ----------
    reconstruct_fn:
        Called as ``fn(X)`` to produce reconstructed output.  Defaults to
        perfect reconstruction (returns X unchanged).
    """
    mock = MagicMock()
    if reconstruct_fn is None:
        mock.predict.side_effect = lambda X, verbose=0: X.astype(np.float64).copy()
    else:
        mock.predict.side_effect = lambda X, verbose=0: reconstruct_fn(X)
    mock.fit.return_value = MagicMock()
    mock.count_params.return_value = 5_000
    mock.save.return_value = None
    return mock


@pytest.fixture
def monitor() -> SensorHealthMonitor:
    """
    SensorHealthMonitor with a tiny real Keras model but mocked predict.

    _calibration_thresholds is set to all-ones so each sensor's health score
    equals ``1 - per_sensor_error`` (clipped to [0, 1]).
    """
    m = SensorHealthMonitor(n_sensors=N_SENSORS, encoding_dim=4)
    # Replace predict with perfect reconstruction so downstream tests control
    # the error analytically.
    m.autoencoder = _make_mock_ae()
    m._calibration_thresholds = np.ones(N_SENSORS, dtype=np.float64)
    return m


@pytest.fixture
def monitor_zero_reconstruction() -> SensorHealthMonitor:
    """Monitor whose autoencoder always predicts zeros (maximum error)."""
    m = SensorHealthMonitor(n_sensors=N_SENSORS, encoding_dim=4)
    m.autoencoder = _make_mock_ae(reconstruct_fn=lambda X: np.zeros_like(X))
    # Thresholds must equal X^2 for a given X to yield score 0.0.
    # Set later in tests where needed.
    m._calibration_thresholds = np.ones(N_SENSORS, dtype=np.float64)
    return m


# ===========================================================================
# Construction
# ===========================================================================


@pytest.mark.unit
class TestConstruction:
    def test_default_n_sensors(self):
        m = SensorHealthMonitor()
        assert m.n_sensors == N_SENSORS

    def test_custom_n_sensors_stored(self):
        m = SensorHealthMonitor(n_sensors=10, encoding_dim=4)
        assert m.n_sensors == 10

    def test_calibration_thresholds_none_before_fit(self):
        m = SensorHealthMonitor(n_sensors=N_SENSORS, encoding_dim=4)
        # A freshly built monitor has no thresholds.
        assert m._calibration_thresholds is None

    def test_invalid_n_sensors_raises(self):
        with pytest.raises(ValueError, match="n_sensors"):
            SensorHealthMonitor(n_sensors=0)

    def test_invalid_encoding_dim_raises(self):
        with pytest.raises(ValueError, match="encoding_dim"):
            SensorHealthMonitor(n_sensors=N_SENSORS, encoding_dim=0)


# ===========================================================================
# Health score bounds
# ===========================================================================


@pytest.mark.unit
class TestHealthScoreBounds:
    def test_scores_in_unit_interval_on_random_input(self, monitor, cmapss_batch):
        scores = monitor.compute_health_scores(cmapss_batch)
        assert scores.shape == (N_SENSORS,)
        assert np.all(scores >= 0.0), f"Negative scores: {scores[scores < 0]}"
        assert np.all(scores <= 1.0), f"Scores above 1: {scores[scores > 1]}"

    def test_scores_in_unit_interval_for_single_reading(self, monitor, cmapss_single):
        scores = monitor.compute_health_scores(cmapss_single)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_scores_type_is_ndarray(self, monitor, cmapss_batch):
        scores = monitor.compute_health_scores(cmapss_batch)
        assert isinstance(scores, np.ndarray)


# ===========================================================================
# Score correctness
# ===========================================================================


@pytest.mark.unit
class TestScoreCorrectness:
    def test_perfect_reconstruction_gives_score_one(self, monitor, cmapss_single):
        """Zero reconstruction error → health score should be 1.0 for each sensor."""
        # monitor.autoencoder.predict returns X unchanged (zero error).
        scores = monitor.compute_health_scores(cmapss_single)
        np.testing.assert_allclose(scores, 1.0, atol=1e-10)

    def test_error_at_threshold_gives_score_zero(self, monitor, cmapss_single):
        """When per-sensor error exactly equals the threshold, score = 0.0."""
        # Override predict to return zeros so error = cmapss_single^2.
        monitor.autoencoder.predict.side_effect = (
            lambda X, verbose=0: np.zeros_like(X)
        )
        # Set threshold to the squared single reading so normalised error = 1.
        monitor._calibration_thresholds = cmapss_single ** 2

        scores = monitor.compute_health_scores(cmapss_single)
        np.testing.assert_allclose(scores, 0.0, atol=1e-10)

    def test_error_exceeding_threshold_clamps_to_zero(self, monitor, cmapss_single):
        """Errors larger than the threshold should still give score 0.0, not negative."""
        monitor.autoencoder.predict.side_effect = (
            lambda X, verbose=0: np.zeros_like(X)
        )
        # Threshold is half the squared error, so ratio > 1 → clamped to 1.
        monitor._calibration_thresholds = (cmapss_single ** 2) * 0.5

        scores = monitor.compute_health_scores(cmapss_single)
        assert np.all(scores >= 0.0)
        np.testing.assert_allclose(scores, 0.0, atol=1e-10)

    def test_per_sensor_independence(self, monitor):
        """Sensors with different errors receive different health scores."""
        reading = np.zeros(N_SENSORS, dtype=np.float64)
        # Set only one sensor to a non-zero value.
        reading[5] = 2.0

        # Autoencoder predicts zeros → only sensor 5 has non-zero error.
        monitor.autoencoder.predict.side_effect = (
            lambda X, verbose=0: np.zeros_like(X)
        )
        # Thresholds all 1.0 → score[5] = 1 - 4.0 < 0, clamped to 0.
        scores = monitor.compute_health_scores(reading)

        assert scores[5] < scores[0], "Degraded sensor should score lower than healthy"
        # Sensors 0–4, 6–20 have zero error → score 1.0.
        healthy_sensors = [i for i in range(N_SENSORS) if i != 5]
        np.testing.assert_allclose(scores[healthy_sensors], 1.0, atol=1e-10)

    def test_temporal_window_averages_errors(self, monitor):
        """
        A 2-D input averages reconstruction error over the time dimension.

        Corrupt half the time steps on sensor 0; the resulting score should
        be between 0 and 1 (not all-zero or all-one).
        """
        data = np.zeros((N_CYCLES, N_SENSORS), dtype=np.float64)
        # Non-zero on half the time steps for sensor 0.
        data[: N_CYCLES // 2, 0] = 1.0

        monitor.autoencoder.predict.side_effect = (
            lambda X, verbose=0: np.zeros_like(X)
        )
        scores = monitor.compute_health_scores(data)

        # Sensor 0 averaged error < 1 (threshold is 1) → score between 0 and 1.
        assert 0.0 < scores[0] < 1.0


# ===========================================================================
# fit() and calibration
# ===========================================================================


@pytest.mark.unit
class TestFit:
    def test_fit_sets_calibration_thresholds(self, rng):
        """fit() must populate _calibration_thresholds from training errors."""
        X_train = rng.standard_normal((80, N_SENSORS)).astype(np.float32)

        m = SensorHealthMonitor(n_sensors=N_SENSORS, encoding_dim=4)
        # Mock the autoencoder so fit() doesn't train and predict is deterministic.
        m.autoencoder = _make_mock_ae()  # perfect reconstruction → zero error

        m.fit(X_train, epochs=1)

        assert m._calibration_thresholds is not None
        assert m._calibration_thresholds.shape == (N_SENSORS,)
        # Perfect reconstruction → training errors are all 0 → floored at 1e-10.
        np.testing.assert_allclose(m._calibration_thresholds, 1e-10)

    def test_fit_thresholds_positive(self, rng):
        """Calibration thresholds must always be strictly positive."""
        X_train = rng.standard_normal((50, N_SENSORS)).astype(np.float32)
        m = SensorHealthMonitor(n_sensors=N_SENSORS, encoding_dim=4)
        m.autoencoder = _make_mock_ae()
        m.fit(X_train, epochs=1)
        assert np.all(m._calibration_thresholds > 0)

    def test_fit_returns_history(self, rng):
        X_train = rng.standard_normal((50, N_SENSORS)).astype(np.float32)
        m = SensorHealthMonitor(n_sensors=N_SENSORS, encoding_dim=4)
        m.autoencoder = _make_mock_ae()
        history = m.fit(X_train, epochs=1)
        assert history is not None

    def test_fit_invalid_calibration_percentile_raises(self, rng):
        X_train = rng.standard_normal((20, N_SENSORS)).astype(np.float32)
        m = SensorHealthMonitor(n_sensors=N_SENSORS, encoding_dim=4)
        m.autoencoder = _make_mock_ae()
        with pytest.raises(ValueError, match="calibration_percentile"):
            m.fit(X_train, epochs=1, calibration_percentile=0.0)


# ===========================================================================
# compute_health_scores without calibration
# ===========================================================================


@pytest.mark.unit
class TestUncalibratedMonitor:
    def test_raises_runtime_error_before_fit(self, cmapss_single):
        m = SensorHealthMonitor(n_sensors=N_SENSORS, encoding_dim=4)
        m.autoencoder = _make_mock_ae()
        # _calibration_thresholds is None → RuntimeError
        with pytest.raises(RuntimeError, match="calibrated"):
            m.compute_health_scores(cmapss_single)


# ===========================================================================
# identify_degraded_sensors
# ===========================================================================


@pytest.mark.unit
class TestIdentifyDegradedSensors:
    def test_returns_indices_below_threshold(self, monitor):
        scores = np.ones(N_SENSORS) * 0.8
        scores[3] = 0.3
        scores[17] = 0.1
        degraded = monitor.identify_degraded_sensors(scores, threshold=0.5)
        assert sorted(degraded) == [3, 17]

    def test_no_degraded_sensors_returns_empty_list(self, monitor):
        scores = np.ones(N_SENSORS)
        degraded = monitor.identify_degraded_sensors(scores, threshold=0.5)
        assert degraded == []

    def test_all_degraded(self, monitor):
        scores = np.zeros(N_SENSORS)
        degraded = monitor.identify_degraded_sensors(scores, threshold=0.5)
        assert sorted(degraded) == list(range(N_SENSORS))

    def test_result_is_sorted(self, monitor):
        scores = np.ones(N_SENSORS)
        scores[[19, 2, 7]] = 0.1
        degraded = monitor.identify_degraded_sensors(scores, threshold=0.5)
        assert degraded == sorted(degraded)

    def test_threshold_out_of_range_raises(self, monitor):
        with pytest.raises(ValueError, match="threshold"):
            monitor.identify_degraded_sensors(np.ones(N_SENSORS), threshold=1.5)


# ===========================================================================
# Input validation
# ===========================================================================


@pytest.mark.unit
class TestInputValidation:
    def test_non_array_raises_type_error(self, monitor):
        with pytest.raises(TypeError, match="numpy.ndarray"):
            monitor.compute_health_scores([[0.0] * N_SENSORS])

    def test_wrong_1d_length_raises(self, monitor):
        with pytest.raises(ValueError, match=str(N_SENSORS)):
            monitor.compute_health_scores(np.ones(10))

    def test_wrong_2d_columns_raises(self, monitor):
        with pytest.raises(ValueError, match=str(N_SENSORS)):
            monitor.compute_health_scores(np.ones((N_CYCLES, 10)))

    def test_3d_input_raises(self, monitor):
        with pytest.raises(ValueError, match="1-D or 2-D"):
            monitor.compute_health_scores(np.ones((2, N_CYCLES, N_SENSORS)))

    def test_nan_input_raises(self, monitor):
        data = np.ones(N_SENSORS)
        data[0] = np.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            monitor.compute_health_scores(data)

    def test_inf_input_raises(self, monitor):
        data = np.ones(N_SENSORS)
        data[5] = np.inf
        with pytest.raises(ValueError, match="NaN or Inf"):
            monitor.compute_health_scores(data)


# ===========================================================================
# Persistence — save / load
# ===========================================================================


@pytest.mark.unit
class TestPersistence:
    def test_save_writes_threshold_file(self, monitor, tmp_path):
        """save() must write calibration_thresholds.npy to the target directory."""
        expected = np.linspace(0.01, 0.21, N_SENSORS)
        monitor._calibration_thresholds = expected

        monitor.save(str(tmp_path / "monitor"))

        npy_path = tmp_path / "monitor" / SensorHealthMonitor._THRESHOLD_FILE
        assert npy_path.exists(), "calibration_thresholds.npy was not created"
        saved = np.load(str(npy_path))
        np.testing.assert_array_equal(saved, expected)

    def test_save_creates_directory(self, monitor, tmp_path):
        target = tmp_path / "deep" / "nested" / "monitor"
        monitor.save(str(target))
        assert target.exists()

    def test_save_without_autoencoder_raises(self):
        m = SensorHealthMonitor.__new__(SensorHealthMonitor)
        m.autoencoder = None
        m._calibration_thresholds = np.ones(N_SENSORS)
        with pytest.raises(RuntimeError, match="No autoencoder"):
            m.save("/tmp/nowhere")

    def test_load_restores_calibration_thresholds(self, monitor, tmp_path):
        """Thresholds written by save() must be restored exactly by load()."""
        expected = np.linspace(0.05, 0.25, N_SENSORS)
        monitor._calibration_thresholds = expected

        monitor_dir = tmp_path / "monitor"
        # Patch autoencoder.save so we don't need Keras to serialize to disk.
        with patch.object(monitor.autoencoder, "save"):
            monitor.save(str(monitor_dir))

        # Load into a fresh monitor using a patched keras.models.load_model.
        m2 = SensorHealthMonitor(n_sensors=N_SENSORS, encoding_dim=4)
        with patch("robustness.sensor_health_monitor.keras") as mock_keras:
            mock_keras.models.load_model.return_value = _make_mock_ae()
            m2.load(str(monitor_dir))

        np.testing.assert_array_equal(m2._calibration_thresholds, expected)

    def test_load_raises_if_directory_missing(self):
        m = SensorHealthMonitor(n_sensors=N_SENSORS, encoding_dim=4)
        with pytest.raises(FileNotFoundError, match="not found"):
            m.load("/this/path/does/not/exist")

    def test_load_raises_if_autoencoder_subdir_missing(self, monitor, tmp_path):
        # Create directory without the autoencoder sub-directory.
        (tmp_path / "monitor").mkdir()
        m = SensorHealthMonitor(n_sensors=N_SENSORS, encoding_dim=4)
        with pytest.raises(FileNotFoundError):
            m.load(str(tmp_path / "monitor"))
