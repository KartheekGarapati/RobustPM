"""
Unit tests for SensorDegradationInjector.

Coverage:
- All four fault modes: gaussian_noise, stuck_at_value, partial_dropout,
  linear_drift.
- Fault is applied only to the specified sensor channels.
- Severity parameter controls the magnitude of corruption.
- Original array is never mutated.
- 1-D (single time-step) inputs are handled transparently.
- Reproducibility with a fixed random seed.
- All validation branches: bad severity, bad indices, wrong dtype/shape.
"""

import numpy as np
import pytest

from robustness.sensor_degradation import SensorDegradationInjector

N_SENSORS = 21
N_CYCLES = 30


# ===========================================================================
# Helpers
# ===========================================================================


def _injector(
    severity: float = 0.5,
    sensors: list | None = None,
    seed: int = 0,
) -> SensorDegradationInjector:
    """Convenience factory with sensible defaults."""
    if sensors is None:
        sensors = [0, 5, 10]
    return SensorDegradationInjector(
        fault_severity=severity,
        affected_sensor_indices=sensors,
        random_seed=seed,
    )


def _unaffected(affected: list) -> list:
    """Return sensor indices NOT in ``affected``."""
    return [i for i in range(N_SENSORS) if i not in affected]


# ===========================================================================
# Construction and validation
# ===========================================================================


@pytest.mark.unit
class TestConstruction:
    def test_default_sensors_are_all_21(self):
        inj = SensorDegradationInjector(fault_severity=0.3)
        assert inj.affected_sensor_indices == list(range(N_SENSORS))

    def test_custom_sensor_indices_stored(self):
        inj = _injector(sensors=[2, 7, 14])
        assert inj.affected_sensor_indices == [2, 7, 14]

    def test_severity_stored(self):
        inj = _injector(severity=0.75)
        assert inj.fault_severity == pytest.approx(0.75)

    def test_severity_boundary_zero(self):
        inj = SensorDegradationInjector(fault_severity=0.0)
        assert inj.fault_severity == 0.0

    def test_severity_boundary_one(self):
        inj = SensorDegradationInjector(fault_severity=1.0)
        assert inj.fault_severity == 1.0

    def test_severity_below_zero_raises(self):
        with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
            SensorDegradationInjector(fault_severity=-0.1)

    def test_severity_above_one_raises(self):
        with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
            SensorDegradationInjector(fault_severity=1.1)

    def test_non_numeric_severity_raises(self):
        with pytest.raises(TypeError, match="numeric"):
            SensorDegradationInjector(fault_severity="high")

    def test_empty_sensor_indices_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            SensorDegradationInjector(fault_severity=0.5, affected_sensor_indices=[])

    def test_negative_sensor_index_raises(self):
        with pytest.raises(ValueError, match="non-negative int"):
            SensorDegradationInjector(fault_severity=0.5, affected_sensor_indices=[-1])

    def test_duplicate_sensor_indices_raises(self):
        with pytest.raises(ValueError, match="duplicate"):
            SensorDegradationInjector(fault_severity=0.5, affected_sensor_indices=[2, 2])

    def test_float_sensor_index_raises(self):
        with pytest.raises(ValueError, match="non-negative int"):
            SensorDegradationInjector(fault_severity=0.5, affected_sensor_indices=[1.0])


# ===========================================================================
# inject() dispatch
# ===========================================================================


@pytest.mark.unit
class TestInjectDispatch:
    @pytest.mark.parametrize(
        "mode",
        ["gaussian_noise", "stuck_at_value", "partial_dropout", "linear_drift"],
    )
    def test_dispatch_all_modes_return_correct_shape(self, cmapss_batch, mode):
        result = _injector().inject(cmapss_batch, mode)
        assert result.shape == cmapss_batch.shape

    @pytest.mark.parametrize(
        "mode",
        ["gaussian_noise", "stuck_at_value", "partial_dropout", "linear_drift"],
    )
    def test_dispatch_returns_copy_not_view(self, cmapss_batch, mode):
        result = _injector().inject(cmapss_batch, mode)
        assert result is not cmapss_batch

    def test_unknown_fault_mode_raises(self, cmapss_batch):
        with pytest.raises(ValueError, match="Unknown fault_mode"):
            _injector().inject(cmapss_batch, "random_teleportation")

    def test_non_array_input_raises(self):
        inj = _injector()
        with pytest.raises(TypeError, match="numpy.ndarray"):
            inj.inject([[0.0] * N_SENSORS] * N_CYCLES, "gaussian_noise")

    def test_3d_input_raises(self, cmapss_batch):
        with pytest.raises(ValueError, match="1-D or 2-D"):
            _injector().inject(cmapss_batch[np.newaxis, :, :], "gaussian_noise")

    def test_index_out_of_bounds_raises(self, cmapss_batch):
        inj = SensorDegradationInjector(
            fault_severity=0.5, affected_sensor_indices=[99]
        )
        with pytest.raises(ValueError, match="references column 99"):
            inj.inject(cmapss_batch, "gaussian_noise")


# ===========================================================================
# Gaussian noise
# ===========================================================================


@pytest.mark.unit
class TestGaussianNoise:
    AFFECTED = [1, 7, 15]

    def test_affected_channels_change(self, cmapss_batch):
        inj = _injector(severity=0.5, sensors=self.AFFECTED, seed=42)
        corrupted = inj.inject_gaussian_noise(cmapss_batch)
        for idx in self.AFFECTED:
            assert not np.allclose(corrupted[:, idx], cmapss_batch[:, idx]), (
                f"Sensor {idx} was not corrupted"
            )

    def test_unaffected_channels_unchanged(self, cmapss_batch):
        inj = _injector(severity=0.8, sensors=self.AFFECTED, seed=0)
        corrupted = inj.inject_gaussian_noise(cmapss_batch)
        for idx in _unaffected(self.AFFECTED):
            np.testing.assert_array_equal(corrupted[:, idx], cmapss_batch[:, idx])

    def test_higher_severity_produces_larger_noise(self, cmapss_batch):
        sensor = 3
        low = _injector(severity=0.1, sensors=[sensor], seed=7).inject_gaussian_noise(cmapss_batch)
        high = _injector(severity=0.9, sensors=[sensor], seed=7).inject_gaussian_noise(cmapss_batch)
        assert np.abs(high[:, sensor] - cmapss_batch[:, sensor]).mean() > np.abs(
            low[:, sensor] - cmapss_batch[:, sensor]
        ).mean()

    def test_zero_severity_produces_no_change(self, cmapss_batch):
        inj = SensorDegradationInjector(
            fault_severity=0.0, affected_sensor_indices=self.AFFECTED, random_seed=0
        )
        corrupted = inj.inject_gaussian_noise(cmapss_batch)
        np.testing.assert_array_almost_equal(corrupted, cmapss_batch)

    def test_original_not_mutated(self, cmapss_batch):
        original = cmapss_batch.copy()
        _injector(sensors=self.AFFECTED, seed=1).inject_gaussian_noise(cmapss_batch)
        np.testing.assert_array_equal(cmapss_batch, original)

    def test_reproducibility_with_seed(self, cmapss_batch):
        a = _injector(seed=42).inject_gaussian_noise(cmapss_batch)
        b = _injector(seed=42).inject_gaussian_noise(cmapss_batch)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_produce_different_output(self, cmapss_batch):
        a = _injector(seed=0).inject_gaussian_noise(cmapss_batch)
        b = _injector(seed=99).inject_gaussian_noise(cmapss_batch)
        assert not np.array_equal(a, b)

    def test_1d_input_returns_1d(self, cmapss_single):
        # inject() squeezes the 2-D internal result back to 1-D.
        corrupted = _injector(seed=0).inject(cmapss_single, "gaussian_noise")
        assert corrupted.shape == cmapss_single.shape


# ===========================================================================
# Stuck-at-value
# ===========================================================================


@pytest.mark.unit
class TestStuckAtValue:
    AFFECTED = [2, 9, 18]

    def _onset(self, n: int, severity: float) -> int:
        return max(0, int(n * (1.0 - severity)))

    def test_values_frozen_after_onset(self, cmapss_batch):
        severity = 0.5
        inj = _injector(severity=severity, sensors=self.AFFECTED)
        corrupted = inj.inject_stuck_at_value(cmapss_batch)
        onset = self._onset(N_CYCLES, severity)
        for idx in self.AFFECTED:
            stuck = cmapss_batch[onset, idx]
            np.testing.assert_array_equal(corrupted[onset:, idx], stuck)

    def test_values_before_onset_unchanged(self, cmapss_batch):
        severity = 0.4
        inj = _injector(severity=severity, sensors=self.AFFECTED)
        corrupted = inj.inject_stuck_at_value(cmapss_batch)
        onset = self._onset(N_CYCLES, severity)
        np.testing.assert_array_equal(
            corrupted[:onset, :], cmapss_batch[:onset, :]
        )

    def test_full_severity_freezes_from_step_zero(self, cmapss_batch):
        inj = _injector(severity=1.0, sensors=[0])
        corrupted = inj.inject_stuck_at_value(cmapss_batch)
        # onset = max(0, int(30*(1-1))) = 0 → entire channel equals first value
        np.testing.assert_array_equal(
            corrupted[:, 0], np.full(N_CYCLES, cmapss_batch[0, 0])
        )

    def test_zero_severity_leaves_data_unchanged(self, cmapss_batch):
        inj = SensorDegradationInjector(
            fault_severity=0.0, affected_sensor_indices=self.AFFECTED
        )
        corrupted = inj.inject_stuck_at_value(cmapss_batch)
        np.testing.assert_array_equal(corrupted, cmapss_batch)

    def test_unaffected_channels_unchanged(self, cmapss_batch):
        inj = _injector(severity=0.6, sensors=self.AFFECTED)
        corrupted = inj.inject_stuck_at_value(cmapss_batch)
        for idx in _unaffected(self.AFFECTED):
            np.testing.assert_array_equal(corrupted[:, idx], cmapss_batch[:, idx])

    def test_original_not_mutated(self, cmapss_batch):
        original = cmapss_batch.copy()
        _injector(sensors=self.AFFECTED).inject_stuck_at_value(cmapss_batch)
        np.testing.assert_array_equal(cmapss_batch, original)

    def test_1d_input_returns_1d(self, cmapss_single):
        result = _injector(severity=1.0, sensors=[0]).inject(cmapss_single, "stuck_at_value")
        assert result.shape == cmapss_single.shape


# ===========================================================================
# Partial dropout
# ===========================================================================


@pytest.mark.unit
class TestPartialDropout:
    AFFECTED = [4, 11, 20]

    def test_zeros_appear_only_on_affected_channels(self, cmapss_batch):
        # Use a batch where values are far from zero (shift to avoid accidental zeros).
        data = cmapss_batch + 10.0
        inj = _injector(severity=0.3, sensors=self.AFFECTED, seed=7)
        corrupted = inj.inject_partial_dropout(data)
        # Zeroed values must come from affected channels only.
        for idx in _unaffected(self.AFFECTED):
            assert np.all(corrupted[:, idx] != 0.0) or np.allclose(
                corrupted[:, idx], data[:, idx]
            ), f"Unaffected sensor {idx} was modified"

    def test_zeroed_fraction_matches_severity(self, cmapss_batch):
        data = cmapss_batch + 10.0  # avoid natural zeros
        severity = 0.4
        sensor = 6
        inj = SensorDegradationInjector(
            fault_severity=severity, affected_sensor_indices=[sensor], random_seed=0
        )
        corrupted = inj.inject_partial_dropout(data)
        n_zeros = int(np.sum(corrupted[:, sensor] == 0.0))
        expected = max(1, int(np.ceil(N_CYCLES * severity)))
        assert n_zeros == expected

    def test_dropped_positions_are_exactly_zero(self, cmapss_batch):
        data = cmapss_batch + 5.0
        inj = _injector(severity=0.5, sensors=[0], seed=3)
        corrupted = inj.inject_partial_dropout(data)
        changed = corrupted[:, 0] != data[:, 0]
        np.testing.assert_array_equal(corrupted[changed, 0], 0.0)

    def test_original_not_mutated(self, cmapss_batch):
        original = cmapss_batch.copy()
        _injector(sensors=self.AFFECTED, seed=2).inject_partial_dropout(cmapss_batch)
        np.testing.assert_array_equal(cmapss_batch, original)

    def test_1d_input_returns_1d(self, cmapss_single):
        result = _injector(severity=0.3, sensors=[0], seed=0).inject(cmapss_single, "partial_dropout")
        assert result.shape == cmapss_single.shape


# ===========================================================================
# Linear drift
# ===========================================================================


@pytest.mark.unit
class TestLinearDrift:
    AFFECTED = [0, 6, 12]

    def test_drift_starts_near_zero_and_grows(self, cmapss_batch):
        inj = _injector(severity=0.8, sensors=self.AFFECTED, seed=0)
        corrupted = inj.inject_linear_drift(cmapss_batch)
        for idx in self.AFFECTED:
            drift = corrupted[:, idx] - cmapss_batch[:, idx]
            # First timestep: t=0 → drift ≈ 0
            assert abs(drift[0]) < 1e-9, (
                f"Sensor {idx}: drift at t=0 should be 0, got {drift[0]}"
            )
            # Final timestep: |drift| > initial (which is 0)
            assert abs(drift[-1]) > abs(drift[0])

    def test_drift_magnitude_scales_with_severity(self, cmapss_batch):
        sensor = 3
        low = _injector(severity=0.1, sensors=[sensor], seed=0).inject_linear_drift(
            cmapss_batch
        )
        high = _injector(severity=0.9, sensors=[sensor], seed=0).inject_linear_drift(
            cmapss_batch
        )
        drift_low = abs(low[-1, sensor] - cmapss_batch[-1, sensor])
        drift_high = abs(high[-1, sensor] - cmapss_batch[-1, sensor])
        assert drift_high > drift_low

    def test_zero_severity_leaves_data_unchanged(self, cmapss_batch):
        inj = SensorDegradationInjector(
            fault_severity=0.0, affected_sensor_indices=self.AFFECTED, random_seed=0
        )
        corrupted = inj.inject_linear_drift(cmapss_batch)
        np.testing.assert_array_almost_equal(corrupted, cmapss_batch)

    def test_unaffected_channels_unchanged(self, cmapss_batch):
        inj = _injector(severity=0.6, sensors=self.AFFECTED, seed=5)
        corrupted = inj.inject_linear_drift(cmapss_batch)
        for idx in _unaffected(self.AFFECTED):
            np.testing.assert_array_equal(corrupted[:, idx], cmapss_batch[:, idx])

    def test_original_not_mutated(self, cmapss_batch):
        original = cmapss_batch.copy()
        _injector(sensors=self.AFFECTED, seed=1).inject_linear_drift(cmapss_batch)
        np.testing.assert_array_equal(cmapss_batch, original)

    def test_1d_input_returns_1d(self, cmapss_single):
        result = _injector(severity=0.5, sensors=[0], seed=0).inject(cmapss_single, "linear_drift")
        assert result.shape == cmapss_single.shape
