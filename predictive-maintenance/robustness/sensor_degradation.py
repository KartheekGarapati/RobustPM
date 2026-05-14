"""
Sensor fault injection for robustness testing of C-MAPSS turbofan sensor data.

Implements four production-relevant fault modes that can be applied to any
subset of the 21 NASA C-MAPSS sensor channels:

- **gaussian_noise** — additive white Gaussian noise scaled to the sensor's
  natural variability.
- **stuck_at_value** — the sensor freezes at its reading at a
  severity-controlled onset time.
- **partial_dropout** — random time-step zeroing at a fraction proportional
  to fault severity.
- **linear_drift** — systematic bias that accumulates linearly over time.

Typical usage::

    injector = SensorDegradationInjector(
        fault_severity=0.6,
        affected_sensor_indices=[1, 7, 10],  # T24, Nf, Ps30
    )
    corrupted = injector.inject(raw_sequence, fault_mode="gaussian_noise")
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Human-readable names for the 21 C-MAPSS sensor channels (zero-indexed).
CMAPSS_SENSOR_NAMES: List[str] = [
    "T2",         # 0  — Total temperature at fan inlet (°R)
    "T24",        # 1  — Total temperature at LPC outlet (°R)
    "T30",        # 2  — Total temperature at HPC outlet (°R)
    "T50",        # 3  — Total temperature at LPT outlet (°R)
    "P2",         # 4  — Pressure at fan inlet (psia)
    "P15",        # 5  — Total pressure in bypass-duct (psia)
    "P30",        # 6  — Total pressure at HPC outlet (psia)
    "Nf",         # 7  — Physical fan speed (rpm)
    "Nc",         # 8  — Physical core speed (rpm)
    "epr",        # 9  — Engine pressure ratio (P50/P2)
    "Ps30",       # 10 — Static pressure at HPC outlet (psia)
    "phi",        # 11 — Ratio of fuel flow to Ps30 (pps/psi)
    "NRf",        # 12 — Corrected fan speed (rpm)
    "NRc",        # 13 — Corrected core speed (rpm)
    "BPR",        # 14 — Bypass Ratio
    "farB",       # 15 — Burner fuel-air ratio
    "htBleed",    # 16 — Bleed Enthalpy
    "Nf_dmd",     # 17 — Demanded fan speed (rpm)
    "PCNfR_dmd",  # 18 — Demanded corrected fan speed (rpm)
    "W31",        # 19 — HPT coolant bleed (lbm/s)
    "W32",        # 20 — LPT coolant bleed (lbm/s)
]

_VALID_FAULT_MODES = frozenset(
    {"gaussian_noise", "stuck_at_value", "partial_dropout", "linear_drift"}
)


class SensorDegradationInjector:
    """
    Injects synthetic sensor faults into NASA C-MAPSS sensor data.

    All fault modes operate on numpy arrays of shape
    ``(n_timesteps, n_sensors)`` and return a corrupted copy without
    modifying the original array.

    Parameters
    ----------
    fault_severity:
        Fault magnitude in **[0.0, 1.0]**.  ``0.0`` produces no corruption;
        ``1.0`` applies maximum corruption for each mode.
    affected_sensor_indices:
        Zero-based column indices into the sensor array.  Defaults to all
        21 C-MAPSS channels when ``None``.
    random_seed:
        Optional integer seed for reproducible fault injection.

    Raises
    ------
    TypeError
        If ``fault_severity`` is not numeric.
    ValueError
        If ``fault_severity`` is outside ``[0, 1]``, if
        ``affected_sensor_indices`` is empty or contains duplicates, or if
        any index is not a non-negative integer.
    """

    N_SENSORS: int = 21

    def __init__(
        self,
        fault_severity: float = 0.5,
        affected_sensor_indices: Optional[List[int]] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        self._validate_severity(fault_severity)
        self.fault_severity = float(fault_severity)

        if affected_sensor_indices is None:
            affected_sensor_indices = list(range(self.N_SENSORS))
        self._validate_sensor_indices(affected_sensor_indices)
        self.affected_sensor_indices: List[int] = list(affected_sensor_indices)

        self._rng = np.random.default_rng(random_seed)

        logger.info(
            "SensorDegradationInjector created | severity=%.2f | "
            "affected_sensors=%s",
            self.fault_severity,
            self.affected_sensor_indices,
        )

    # ------------------------------------------------------------------
    # Public dispatch
    # ------------------------------------------------------------------

    def inject(self, data: np.ndarray, fault_mode: str) -> np.ndarray:
        """
        Apply a fault mode to ``data``.

        Parameters
        ----------
        data:
            Sensor readings of shape ``(n_timesteps, n_sensors)`` or
            ``(n_sensors,)`` for a single time-step.
        fault_mode:
            One of ``"gaussian_noise"``, ``"stuck_at_value"``,
            ``"partial_dropout"``, or ``"linear_drift"``.

        Returns
        -------
        np.ndarray
            Corrupted copy of ``data``; shape identical to the input.

        Raises
        ------
        ValueError
            On unrecognised ``fault_mode`` or incompatible data shape.
        TypeError
            If ``data`` is not a ``numpy.ndarray``.
        """
        if fault_mode not in _VALID_FAULT_MODES:
            raise ValueError(
                f"Unknown fault_mode '{fault_mode}'. "
                f"Valid options: {sorted(_VALID_FAULT_MODES)}"
            )

        data_2d = self._validate_and_coerce(data)

        dispatch = {
            "gaussian_noise": self.inject_gaussian_noise,
            "stuck_at_value": self.inject_stuck_at_value,
            "partial_dropout": self.inject_partial_dropout,
            "linear_drift": self.inject_linear_drift,
        }
        corrupted = dispatch[fault_mode](data_2d)

        logger.debug(
            "Injected '%s' (severity=%.2f) into sensors %s",
            fault_mode,
            self.fault_severity,
            self.affected_sensor_indices,
        )
        # Restore original shape if the caller passed a 1-D array.
        return corrupted.squeeze(0) if data.ndim == 1 else corrupted

    # ------------------------------------------------------------------
    # Fault modes
    # ------------------------------------------------------------------

    def inject_gaussian_noise(self, data: np.ndarray) -> np.ndarray:
        """
        Add zero-mean Gaussian noise to the affected sensor channels.

        Noise standard deviation equals ``fault_severity × channel_std``,
        so the amplitude is proportional to each sensor's natural variability.
        Constant channels (std ≈ 0) receive noise with std = 1.0.

        Parameters
        ----------
        data:
            Shape ``(n_timesteps, n_sensors)``.

        Returns
        -------
        np.ndarray
            Corrupted copy; shape unchanged.
        """
        data = self._validate_and_coerce(data)
        corrupted = data.copy()

        for idx in self.affected_sensor_indices:
            channel_std = float(np.std(data[:, idx]))
            if channel_std < 1e-8:
                channel_std = 1.0
            noise = self._rng.normal(
                loc=0.0,
                scale=self.fault_severity * channel_std,
                size=data.shape[0],
            )
            corrupted[:, idx] += noise

        return corrupted

    def inject_stuck_at_value(self, data: np.ndarray) -> np.ndarray:
        """
        Freeze affected sensors at a severity-controlled onset time.

        The onset time-step is ``floor(n_timesteps × (1 − fault_severity))``.
        At severity 1.0 the sensor freezes from time-step 0; at 0.5 it
        freezes from the midpoint.

        Parameters
        ----------
        data:
            Shape ``(n_timesteps, n_sensors)``.

        Returns
        -------
        np.ndarray
            Corrupted copy; shape unchanged.
        """
        data = self._validate_and_coerce(data)
        n_timesteps = data.shape[0]
        corrupted = data.copy()

        onset = max(0, int(n_timesteps * (1.0 - self.fault_severity)))

        for idx in self.affected_sensor_indices:
            if onset >= n_timesteps:
                # severity so low that the onset is past the sequence end; no-op.
                continue
            stuck_value = float(data[onset, idx])
            corrupted[onset:, idx] = stuck_value

        return corrupted

    def inject_partial_dropout(self, data: np.ndarray) -> np.ndarray:
        """
        Randomly zero out readings in affected sensor channels.

        The fraction of zeroed time-steps equals ``fault_severity``.
        At severity 1.0 every reading is zeroed; at 0.3, 30 % of readings
        are zeroed.

        Parameters
        ----------
        data:
            Shape ``(n_timesteps, n_sensors)``.

        Returns
        -------
        np.ndarray
            Corrupted copy; shape unchanged.
        """
        data = self._validate_and_coerce(data)
        n_timesteps = data.shape[0]
        corrupted = data.copy()

        n_dropped = max(1, int(np.ceil(n_timesteps * self.fault_severity)))

        for idx in self.affected_sensor_indices:
            drop_indices = self._rng.choice(
                n_timesteps, size=min(n_dropped, n_timesteps), replace=False
            )
            corrupted[drop_indices, idx] = 0.0

        return corrupted

    def inject_linear_drift(self, data: np.ndarray) -> np.ndarray:
        """
        Add a linearly increasing bias to affected sensor channels.

        The total drift over the sequence equals
        ``fault_severity × channel_std``.  The direction (positive or
        negative) is chosen independently per sensor to avoid all channels
        drifting in the same direction.

        Parameters
        ----------
        data:
            Shape ``(n_timesteps, n_sensors)``.

        Returns
        -------
        np.ndarray
            Corrupted copy; shape unchanged.
        """
        data = self._validate_and_coerce(data)
        n_timesteps = data.shape[0]
        corrupted = data.copy()

        # Ramp from 0 → 1 across the sequence length.
        t = np.linspace(0.0, 1.0, n_timesteps)

        for idx in self.affected_sensor_indices:
            channel_std = float(np.std(data[:, idx]))
            if channel_std < 1e-8:
                channel_std = 1.0
            drift_amplitude = self.fault_severity * channel_std
            direction = self._rng.choice([-1.0, 1.0])
            corrupted[:, idx] += direction * drift_amplitude * t

        return corrupted

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_and_coerce(self, data: np.ndarray) -> np.ndarray:
        """Return a float64 2-D array, promoting 1-D inputs and validating shape."""
        if not isinstance(data, np.ndarray):
            raise TypeError(
                f"data must be a numpy.ndarray, got {type(data).__name__}"
            )
        data = data.astype(np.float64, copy=False)
        if data.ndim == 1:
            data = data[np.newaxis, :]  # (n_sensors,) → (1, n_sensors)
        if data.ndim != 2:
            raise ValueError(
                f"data must be 1-D or 2-D, got shape {data.shape}"
            )
        if self.affected_sensor_indices:
            max_idx = max(self.affected_sensor_indices)
            if data.shape[1] <= max_idx:
                raise ValueError(
                    f"data has {data.shape[1]} sensor columns but "
                    f"affected_sensor_indices references column {max_idx}"
                )
        return data

    @staticmethod
    def _validate_severity(severity: float) -> None:
        if not isinstance(severity, (int, float)):
            raise TypeError(
                f"fault_severity must be numeric, got {type(severity).__name__}"
            )
        if not 0.0 <= float(severity) <= 1.0:
            raise ValueError(
                f"fault_severity must be in [0.0, 1.0], got {severity}"
            )

    @staticmethod
    def _validate_sensor_indices(indices: List[int]) -> None:
        if not indices:
            raise ValueError("affected_sensor_indices must not be empty")
        for idx in indices:
            if not isinstance(idx, int) or idx < 0:
                raise ValueError(
                    f"Each sensor index must be a non-negative int, got {idx!r}"
                )
        if len(set(indices)) != len(indices):
            raise ValueError(
                "affected_sensor_indices contains duplicate entries"
            )
