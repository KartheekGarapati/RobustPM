"""
Unit tests for RobustInferenceEngine.

TensorFlow is required for the ``_TF_AVAILABLE`` flag inside the module.
The LSTM model is always replaced by a ``MagicMock`` so no GPU/TPU is needed
and tests run in milliseconds.

Coverage:
- Channel weights: degraded sensors get lower weights; uniform health → weight 1.
- min_weight floor prevents channels from being fully zeroed.
- Weights are mean-normalised (mean = 1.0) regardless of health profile.
- apply_weights multiplies sensor channels element-wise across the time axis.
- predict_rul returns all documented result keys.
- RUL is clipped to rul_clip_max and floored at 0.
- Degraded sensors in the result match the expected indices and count.
- Health-status string mapping covers all four thresholds.
- Without explicit health_scores and without a health_monitor, uniform
  weights (all-1.0) are applied.
- When a SensorHealthMonitor is attached, its compute_health_scores() is
  called automatically.
- Explicit health_scores override the attached monitor.
- from_model_manager() retrieves the LSTM via get_model("lstm").
- from_model_manager() raises RuntimeError when the model is not loaded.
- Invalid constructor arguments raise ValueError / ImportError.
- Invalid sequence shapes and out-of-range health_scores raise ValueError.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

# Skip the entire module when TensorFlow is unavailable.
# We catch ValueError too because a numpy/h5py ABI mismatch raises ValueError
# rather than ImportError when TF tries to load h5py.
try:
    import tensorflow as _tf  # noqa: F401
except (ImportError, ValueError) as _tf_err:
    pytest.skip(
        f"TensorFlow not available ({_tf_err}); skipping RobustInferenceEngine tests.",
        allow_module_level=True,
    )

from robustness.robust_inference import RobustInferenceEngine

N_SENSORS = 21
N_CYCLES = 30

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def mock_lstm() -> MagicMock:
    """Mock LSTM model that predicts RUL = 75.0 cycles."""
    model = MagicMock()
    model.predict.return_value = np.array([[75.0]])
    return model


@pytest.fixture
def engine(mock_lstm) -> RobustInferenceEngine:
    """RobustInferenceEngine with a mock LSTM and no health monitor."""
    return RobustInferenceEngine(
        lstm_model=mock_lstm,
        n_sensors=N_SENSORS,
        min_weight=0.1,
        rul_clip_max=200.0,
    )


@pytest.fixture
def healthy_scores() -> np.ndarray:
    """All sensors fully healthy (score 1.0)."""
    return np.ones(N_SENSORS, dtype=np.float64)


@pytest.fixture
def degraded_scores() -> np.ndarray:
    """Sensors 3 and 15 degraded; rest healthy."""
    scores = np.ones(N_SENSORS, dtype=np.float64)
    scores[3] = 0.2
    scores[15] = 0.05
    return scores


# ===========================================================================
# Construction
# ===========================================================================


@pytest.mark.unit
class TestConstruction:
    def test_stores_n_sensors(self, mock_lstm):
        e = RobustInferenceEngine(lstm_model=mock_lstm, n_sensors=N_SENSORS)
        assert e.n_sensors == N_SENSORS

    def test_stores_min_weight(self, mock_lstm):
        e = RobustInferenceEngine(lstm_model=mock_lstm, n_sensors=N_SENSORS, min_weight=0.2)
        assert e.min_weight == pytest.approx(0.2)

    def test_stores_rul_clip_max(self, mock_lstm):
        e = RobustInferenceEngine(lstm_model=mock_lstm, n_sensors=N_SENSORS, rul_clip_max=150.0)
        assert e.rul_clip_max == pytest.approx(150.0)

    def test_none_lstm_raises(self):
        with pytest.raises(ValueError, match="must not be None"):
            RobustInferenceEngine(lstm_model=None, n_sensors=N_SENSORS)

    def test_min_weight_above_one_raises(self, mock_lstm):
        with pytest.raises(ValueError, match="min_weight"):
            RobustInferenceEngine(lstm_model=mock_lstm, n_sensors=N_SENSORS, min_weight=1.5)

    def test_negative_min_weight_raises(self, mock_lstm):
        with pytest.raises(ValueError, match="min_weight"):
            RobustInferenceEngine(lstm_model=mock_lstm, n_sensors=N_SENSORS, min_weight=-0.1)

    def test_zero_n_sensors_raises(self, mock_lstm):
        with pytest.raises(ValueError, match="n_sensors"):
            RobustInferenceEngine(lstm_model=mock_lstm, n_sensors=0)

    def test_nonpositive_rul_clip_max_raises(self, mock_lstm):
        with pytest.raises(ValueError, match="rul_clip_max"):
            RobustInferenceEngine(lstm_model=mock_lstm, n_sensors=N_SENSORS, rul_clip_max=0.0)

    def test_importerror_when_tf_unavailable(self, mock_lstm):
        with patch("robustness.robust_inference._TF_AVAILABLE", False):
            with pytest.raises(ImportError, match="TensorFlow"):
                RobustInferenceEngine(lstm_model=mock_lstm, n_sensors=N_SENSORS)


# ===========================================================================
# compute_channel_weights
# ===========================================================================


@pytest.mark.unit
class TestComputeChannelWeights:
    def test_degraded_sensor_receives_lower_weight(self, engine, degraded_scores):
        weights = engine.compute_channel_weights(degraded_scores)
        assert weights[3] < weights[0], "Sensor 3 (score=0.2) should weigh less than sensor 0 (score=1.0)"
        assert weights[15] < weights[0], "Sensor 15 (score=0.05) should weigh less than sensor 0"
        assert weights[15] < weights[3], "More degraded sensor should have lower weight"

    def test_uniform_health_gives_uniform_weights(self, engine, healthy_scores):
        weights = engine.compute_channel_weights(healthy_scores)
        np.testing.assert_allclose(weights, 1.0, atol=1e-10)

    def test_weights_are_mean_normalised(self, engine, degraded_scores):
        """Mean of the channel weights must equal 1.0."""
        weights = engine.compute_channel_weights(degraded_scores)
        np.testing.assert_allclose(weights.mean(), 1.0, atol=1e-10)

    def test_weights_mean_normalised_on_random_scores(self, engine, rng):
        scores = rng.uniform(0.2, 1.0, N_SENSORS)
        weights = engine.compute_channel_weights(scores)
        np.testing.assert_allclose(weights.mean(), 1.0, atol=1e-10)

    def test_min_weight_floor_prevents_zero(self, engine):
        """Sensor with score 0.0 must still receive min_weight > 0."""
        scores = np.ones(N_SENSORS)
        scores[0] = 0.0
        weights = engine.compute_channel_weights(scores)
        assert weights[0] > 0.0, "Zero health score should not produce zero weight"

    def test_min_weight_floor_keeps_weight_below_healthy(self, engine):
        scores = np.ones(N_SENSORS)
        scores[0] = 0.0
        weights = engine.compute_channel_weights(scores)
        # Floored sensor must weigh less than a fully healthy sensor.
        assert weights[0] < weights[1]

    def test_wrong_shape_raises(self, engine):
        with pytest.raises(ValueError, match=r"must have shape"):
            engine.compute_channel_weights(np.ones(10))


# ===========================================================================
# apply_weights
# ===========================================================================


@pytest.mark.unit
class TestApplyWeights:
    def test_output_shape_preserved(self, engine, cmapss_batch):
        weights = np.ones(N_SENSORS)
        weighted = engine.apply_weights(cmapss_batch.astype(np.float32), weights)
        assert weighted.shape == cmapss_batch.shape

    def test_unit_weights_are_identity(self, engine, cmapss_batch):
        weights = np.ones(N_SENSORS)
        weighted = engine.apply_weights(cmapss_batch.astype(np.float32), weights)
        np.testing.assert_allclose(weighted, cmapss_batch, rtol=1e-5)

    def test_halved_weight_halves_channel(self, engine, cmapss_batch):
        weights = np.ones(N_SENSORS)
        weights[7] = 0.5
        weighted = engine.apply_weights(cmapss_batch.astype(np.float32), weights)
        np.testing.assert_allclose(
            weighted[:, 7], cmapss_batch[:, 7] * 0.5, rtol=1e-5
        )

    def test_other_channels_unaffected(self, engine, cmapss_batch):
        weights = np.ones(N_SENSORS)
        weights[7] = 0.5
        weighted = engine.apply_weights(cmapss_batch.astype(np.float32), weights)
        for idx in range(N_SENSORS):
            if idx == 7:
                continue
            np.testing.assert_allclose(
                weighted[:, idx], cmapss_batch[:, idx], rtol=1e-5
            )

    def test_broadcast_applies_across_all_timesteps(self, engine, cmapss_batch):
        """Each time step should receive the same per-channel weight."""
        weights = np.arange(1, N_SENSORS + 1, dtype=np.float64)
        weighted = engine.apply_weights(cmapss_batch.astype(np.float64), weights)
        for t in range(N_CYCLES):
            np.testing.assert_allclose(
                weighted[t, :], cmapss_batch[t, :] * weights, rtol=1e-10
            )

    def test_wrong_weight_shape_raises(self, engine, cmapss_batch):
        with pytest.raises(ValueError, match="must have shape"):
            engine.apply_weights(cmapss_batch.astype(np.float32), np.ones(10))


# ===========================================================================
# predict_rul — output structure and values
# ===========================================================================


@pytest.mark.unit
class TestPredictRUL:
    _REQUIRED_KEYS = {
        "rul_cycles",
        "rul_hours",
        "health_status",
        "channel_weights",
        "health_scores",
        "degraded_sensors",
        "n_degraded",
        "latency_ms",
    }

    def test_result_contains_all_required_keys(self, engine, cmapss_batch):
        result = engine.predict_rul(cmapss_batch.astype(np.float32))
        assert self._REQUIRED_KEYS.issubset(result.keys())

    def test_rul_cycles_equals_mock_output(self, engine, cmapss_batch):
        result = engine.predict_rul(cmapss_batch.astype(np.float32))
        assert result["rul_cycles"] == pytest.approx(75.0)

    def test_rul_hours_is_half_cycles(self, engine, cmapss_batch):
        result = engine.predict_rul(cmapss_batch.astype(np.float32))
        assert result["rul_hours"] == pytest.approx(result["rul_cycles"] * 0.5)

    def test_rul_clipped_to_max(self, mock_lstm, cmapss_batch):
        mock_lstm.predict.return_value = np.array([[999.0]])
        e = RobustInferenceEngine(lstm_model=mock_lstm, n_sensors=N_SENSORS, rul_clip_max=200.0)
        result = e.predict_rul(cmapss_batch.astype(np.float32))
        assert result["rul_cycles"] == pytest.approx(200.0)

    def test_negative_rul_clipped_to_zero(self, mock_lstm, cmapss_batch):
        mock_lstm.predict.return_value = np.array([[-50.0]])
        e = RobustInferenceEngine(lstm_model=mock_lstm, n_sensors=N_SENSORS)
        result = e.predict_rul(cmapss_batch.astype(np.float32))
        assert result["rul_cycles"] == pytest.approx(0.0)

    def test_channel_weights_length_equals_n_sensors(self, engine, cmapss_batch):
        result = engine.predict_rul(cmapss_batch.astype(np.float32))
        assert len(result["channel_weights"]) == N_SENSORS

    def test_health_scores_length_equals_n_sensors(self, engine, cmapss_batch):
        result = engine.predict_rul(cmapss_batch.astype(np.float32))
        assert len(result["health_scores"]) == N_SENSORS

    def test_latency_ms_is_positive(self, engine, cmapss_batch):
        result = engine.predict_rul(cmapss_batch.astype(np.float32))
        assert result["latency_ms"] > 0.0

    def test_lstm_predict_called_once(self, engine, mock_lstm, cmapss_batch):
        engine.predict_rul(cmapss_batch.astype(np.float32))
        assert mock_lstm.predict.call_count == 1


# ===========================================================================
# predict_rul — degraded sensor identification
# ===========================================================================


@pytest.mark.unit
class TestDegradedSensorIdentification:
    def test_degraded_sensors_correctly_flagged(self, engine, cmapss_batch, degraded_scores):
        result = engine.predict_rul(
            cmapss_batch.astype(np.float32),
            health_scores=degraded_scores,
            degraded_threshold=0.5,
        )
        assert 3 in result["degraded_sensors"]
        assert 15 in result["degraded_sensors"]

    def test_degraded_sensors_count(self, engine, cmapss_batch, degraded_scores):
        result = engine.predict_rul(
            cmapss_batch.astype(np.float32),
            health_scores=degraded_scores,
            degraded_threshold=0.5,
        )
        assert result["n_degraded"] == 2

    def test_healthy_sensors_not_flagged(self, engine, cmapss_batch, degraded_scores):
        result = engine.predict_rul(
            cmapss_batch.astype(np.float32),
            health_scores=degraded_scores,
            degraded_threshold=0.5,
        )
        healthy = [i for i in range(N_SENSORS) if i not in [3, 15]]
        for idx in healthy:
            assert idx not in result["degraded_sensors"]

    def test_degraded_sensors_list_is_sorted(self, engine, cmapss_batch, degraded_scores):
        result = engine.predict_rul(
            cmapss_batch.astype(np.float32),
            health_scores=degraded_scores,
        )
        assert result["degraded_sensors"] == sorted(result["degraded_sensors"])

    def test_no_degraded_sensors_when_all_healthy(self, engine, cmapss_batch, healthy_scores):
        result = engine.predict_rul(
            cmapss_batch.astype(np.float32),
            health_scores=healthy_scores,
            degraded_threshold=0.5,
        )
        assert result["degraded_sensors"] == []
        assert result["n_degraded"] == 0

    def test_high_threshold_flags_more_sensors(self, engine, cmapss_batch):
        scores = np.full(N_SENSORS, 0.6)
        scores[0] = 0.3

        low_thresh = engine.predict_rul(
            cmapss_batch.astype(np.float32), health_scores=scores, degraded_threshold=0.2
        )
        high_thresh = engine.predict_rul(
            cmapss_batch.astype(np.float32), health_scores=scores, degraded_threshold=0.7
        )
        assert high_thresh["n_degraded"] >= low_thresh["n_degraded"]


# ===========================================================================
# predict_rul — health status mapping
# ===========================================================================


@pytest.mark.unit
class TestHealthStatusMapping:
    @pytest.mark.parametrize(
        "rul, expected_status",
        [
            (150.0, "healthy"),
            (100.0, "healthy"),  # boundary: ≥ 100
            (99.9,  "warning"),
            (75.0,  "warning"),
            (50.0,  "warning"),   # boundary: ≥ 50
            (49.9,  "critical"),
            (25.0,  "critical"),
            (10.0,  "critical"),  # boundary: ≥ 10
            (9.9,   "imminent_failure"),
            (0.0,   "imminent_failure"),
        ],
    )
    def test_rul_to_health_status(self, engine, rul, expected_status):
        assert engine._rul_to_health_status(rul) == expected_status

    def test_health_status_in_result(self, mock_lstm, cmapss_batch):
        mock_lstm.predict.return_value = np.array([[75.0]])  # warning range
        e = RobustInferenceEngine(lstm_model=mock_lstm, n_sensors=N_SENSORS)
        result = e.predict_rul(cmapss_batch.astype(np.float32))
        assert result["health_status"] == "warning"


# ===========================================================================
# predict_rul — health score resolution (uniform, monitor, explicit)
# ===========================================================================


@pytest.mark.unit
class TestHealthScoreResolution:
    def test_no_scores_no_monitor_applies_uniform_weights(self, engine, cmapss_batch):
        """Without scores or a monitor, all channels should be weighted equally."""
        result = engine.predict_rul(cmapss_batch.astype(np.float32), health_scores=None)
        np.testing.assert_array_equal(result["health_scores"], [1.0] * N_SENSORS)
        np.testing.assert_allclose(result["channel_weights"], [1.0] * N_SENSORS, atol=1e-10)

    def test_health_monitor_called_when_no_explicit_scores(self, mock_lstm, cmapss_batch):
        """When health_scores is None, the monitor's compute_health_scores is called."""
        mock_monitor = MagicMock()
        expected_scores = np.full(N_SENSORS, 0.8)
        mock_monitor.compute_health_scores.return_value = expected_scores

        e = RobustInferenceEngine(
            lstm_model=mock_lstm,
            health_monitor=mock_monitor,
            n_sensors=N_SENSORS,
        )
        result = e.predict_rul(cmapss_batch.astype(np.float32), health_scores=None)

        mock_monitor.compute_health_scores.assert_called_once()
        np.testing.assert_array_equal(result["health_scores"], expected_scores.tolist())

    def test_explicit_scores_override_monitor(self, mock_lstm, cmapss_batch):
        """Explicitly supplied health_scores must take priority over the monitor."""
        mock_monitor = MagicMock()
        mock_monitor.compute_health_scores.return_value = np.full(N_SENSORS, 0.3)

        explicit = np.full(N_SENSORS, 0.9)
        e = RobustInferenceEngine(
            lstm_model=mock_lstm,
            health_monitor=mock_monitor,
            n_sensors=N_SENSORS,
        )
        result = e.predict_rul(
            cmapss_batch.astype(np.float32), health_scores=explicit
        )

        mock_monitor.compute_health_scores.assert_not_called()
        np.testing.assert_array_equal(result["health_scores"], explicit.tolist())

    def test_monitor_failure_falls_back_to_uniform(self, mock_lstm, cmapss_batch):
        """If the monitor raises, fall back to uniform weights without crashing."""
        mock_monitor = MagicMock()
        mock_monitor.compute_health_scores.side_effect = RuntimeError("monitor down")

        e = RobustInferenceEngine(
            lstm_model=mock_lstm,
            health_monitor=mock_monitor,
            n_sensors=N_SENSORS,
        )
        result = e.predict_rul(cmapss_batch.astype(np.float32), health_scores=None)

        # Fallback: uniform weights → scores all 1.0
        np.testing.assert_array_equal(result["health_scores"], [1.0] * N_SENSORS)


# ===========================================================================
# predict_rul — input validation
# ===========================================================================


@pytest.mark.unit
class TestPredictRULValidation:
    def test_non_array_sequence_raises(self, engine):
        with pytest.raises(TypeError, match="numpy.ndarray"):
            engine.predict_rul([[0.0] * N_SENSORS] * N_CYCLES)

    def test_1d_sequence_raises(self, engine):
        with pytest.raises(ValueError, match="2-D"):
            engine.predict_rul(np.ones(N_SENSORS, dtype=np.float32))

    def test_wrong_sensor_count_raises(self, engine):
        with pytest.raises(ValueError, match="sensor columns"):
            engine.predict_rul(np.ones((N_CYCLES, 10), dtype=np.float32))

    def test_nan_in_sequence_raises(self, engine, cmapss_batch):
        bad = cmapss_batch.astype(np.float32)
        bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            engine.predict_rul(bad)

    def test_health_scores_wrong_shape_raises(self, engine, cmapss_batch):
        with pytest.raises(ValueError, match=r"must have shape"):
            engine.predict_rul(
                cmapss_batch.astype(np.float32), health_scores=np.ones(10)
            )

    def test_health_scores_above_one_raises(self, engine, cmapss_batch):
        scores = np.ones(N_SENSORS)
        scores[0] = 1.1
        with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
            engine.predict_rul(cmapss_batch.astype(np.float32), health_scores=scores)

    def test_health_scores_below_zero_raises(self, engine, cmapss_batch):
        scores = np.ones(N_SENSORS)
        scores[2] = -0.1
        with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
            engine.predict_rul(cmapss_batch.astype(np.float32), health_scores=scores)

    def test_lstm_error_raises_runtime_error(self, mock_lstm, cmapss_batch):
        mock_lstm.predict.side_effect = Exception("GPU OOM")
        e = RobustInferenceEngine(lstm_model=mock_lstm, n_sensors=N_SENSORS)
        with pytest.raises(RuntimeError, match="LSTM prediction failed"):
            e.predict_rul(cmapss_batch.astype(np.float32))


# ===========================================================================
# from_model_manager
# ===========================================================================


@pytest.mark.unit
class TestFromModelManager:
    def test_retrieves_lstm_from_manager(self, mock_lstm):
        mock_mm = MagicMock()
        mock_mm.get_model.return_value = mock_lstm

        e = RobustInferenceEngine.from_model_manager(mock_mm)

        mock_mm.get_model.assert_called_once_with("lstm")
        assert e.lstm_model is mock_lstm

    def test_raises_if_lstm_not_loaded(self):
        mock_mm = MagicMock()
        mock_mm.get_model.return_value = None

        with pytest.raises(RuntimeError, match="not loaded"):
            RobustInferenceEngine.from_model_manager(mock_mm)

    def test_kwargs_forwarded_to_engine(self, mock_lstm):
        mock_mm = MagicMock()
        mock_mm.get_model.return_value = mock_lstm

        e = RobustInferenceEngine.from_model_manager(mock_mm, min_weight=0.2, n_sensors=N_SENSORS)

        assert e.min_weight == pytest.approx(0.2)
        assert e.n_sensors == N_SENSORS

    def test_health_monitor_attached(self, mock_lstm):
        mock_mm = MagicMock()
        mock_mm.get_model.return_value = mock_lstm
        mock_monitor = MagicMock()

        e = RobustInferenceEngine.from_model_manager(mock_mm, health_monitor=mock_monitor)

        assert e.health_monitor is mock_monitor
