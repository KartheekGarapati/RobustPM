"""
Robust LSTM RUL inference with dynamic sensor-channel weighting.

Degraded sensors are identified via per-sensor health scores and
down-weighted before the sequence reaches the LSTM predictor, preventing
faulty readings from distorting the RUL estimate without requiring a model
retrain.

Weighting strategy
------------------
Each channel weight ``w[i]`` is computed from health score ``h[i]``::

    raw_weight[i] = max(h[i], min_weight)

Weights are then mean-normalised so the overall input magnitude is
preserved relative to the unweighted case::

    w_norm[i] = raw_weight[i] × n_sensors / Σ raw_weight

The weighted input sequence is::

    weighted_sequence[:, i] = raw_sequence[:, i] × w_norm[i]

FastAPI integration
-------------------
Add the ``robust_router`` to the FastAPI app in
``inference_service/api/main.py``::

    from robustness.robust_inference import robust_router
    app.include_router(robust_router, prefix="/predict", tags=["robust"])

This exposes a new endpoint ``POST /predict/robust_rul`` that accepts
raw 21-sensor sequences alongside optional pre-computed health scores.

Standalone usage::

    engine = RobustInferenceEngine(
        lstm_model=model_manager.get_model("lstm"),
        health_monitor=monitor,          # optional
    )
    result = engine.predict_rul(sequence, health_scores=scores)
    # result["rul_cycles"] → predicted RUL
    # result["degraded_sensors"] → [7, 10] (example)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf

    _TF_AVAILABLE = True
except (ImportError, ValueError):
    # ValueError is raised when h5py has a binary incompatibility with numpy.
    tf = None  # type: ignore[assignment]
    _TF_AVAILABLE = False
    logger.warning(
        "TensorFlow not available — RobustInferenceEngine requires TensorFlow"
    )

try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel, Field

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

from .sensor_health_monitor import SensorHealthMonitor


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------


class RobustInferenceEngine:
    """
    LSTM RUL predictor with health-score-based sensor-channel weighting.

    Parameters
    ----------
    lstm_model:
        A loaded Keras model that accepts inputs of shape
        ``(1, seq_len, n_sensors)`` and returns a scalar RUL estimate.
    health_monitor:
        Optional :class:`~robustness.sensor_health_monitor.SensorHealthMonitor`.
        When provided and ``health_scores`` is not supplied to
        :meth:`predict_rul`, health scores are computed automatically
        from the raw input sequence.  If neither is available, uniform
        weights (no-op) are applied.
    min_weight:
        Floor applied to channel weights before normalisation.  Prevents
        a fully degraded sensor from being zeroed out, which could push
        the input distribution out of the model's training domain.
        Default ``0.1`` (10 % of nominal weight).
    n_sensors:
        Number of raw sensor channels in the input sequence.
        Must equal the last dimension of arrays passed to
        :meth:`predict_rul`.
    rul_clip_max:
        Hard upper bound on the predicted RUL (engine cycles).
        Clips physiologically implausible outputs from the LSTM.
        Default ``200``.

    Raises
    ------
    ImportError
        If TensorFlow is not installed.
    ValueError
        If ``lstm_model`` is ``None`` or ``min_weight`` is outside
        ``[0, 1]``.
    """

    def __init__(
        self,
        lstm_model: Any,
        health_monitor: Optional[SensorHealthMonitor] = None,
        min_weight: float = 0.1,
        n_sensors: int = 21,
        rul_clip_max: float = 200.0,
    ) -> None:
        if not _TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is required for RobustInferenceEngine. "
                "Install it with: pip install tensorflow"
            )
        if lstm_model is None:
            raise ValueError("lstm_model must not be None")
        if not 0.0 <= min_weight <= 1.0:
            raise ValueError(
                f"min_weight must be in [0.0, 1.0], got {min_weight}"
            )
        if n_sensors < 1:
            raise ValueError(f"n_sensors must be >= 1, got {n_sensors}")
        if rul_clip_max <= 0:
            raise ValueError(
                f"rul_clip_max must be positive, got {rul_clip_max}"
            )

        self.lstm_model = lstm_model
        self.health_monitor = health_monitor
        self.min_weight = float(min_weight)
        self.n_sensors = n_sensors
        self.rul_clip_max = float(rul_clip_max)

        logger.info(
            "RobustInferenceEngine initialised | n_sensors=%d | "
            "min_weight=%.2f | rul_clip_max=%.0f",
            self.n_sensors,
            self.min_weight,
            self.rul_clip_max,
        )

    # ------------------------------------------------------------------
    # Main public interface
    # ------------------------------------------------------------------

    def predict_rul(
        self,
        sequence: np.ndarray,
        health_scores: Optional[np.ndarray] = None,
        degraded_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Predict RUL with dynamic per-sensor channel weighting.

        Parameters
        ----------
        sequence:
            Normalised sensor readings of shape ``(seq_len, n_sensors)``.
            Values should be z-score or min-max scaled to match the LSTM's
            training distribution before being passed here.
        health_scores:
            Per-sensor health scores in ``[0.0, 1.0]``, shape
            ``(n_sensors,)``.  When ``None`` and a :attr:`health_monitor`
            is attached, scores are computed from ``sequence`` automatically.
            When neither is available, all channels receive weight 1.0.
        degraded_threshold:
            Sensors with a health score below this value are reported in
            ``result["degraded_sensors"]``.  Default ``0.5``.

        Returns
        -------
        dict
            ``rul_cycles`` (float)
                Predicted remaining useful life in engine cycles.
            ``rul_hours`` (float)
                Approximate RUL in hours (1 cycle ≈ 0.5 h).
            ``health_status`` (str)
                One of ``"healthy"``, ``"warning"``, ``"critical"``,
                or ``"imminent_failure"``.
            ``channel_weights`` (list[float])
                Normalised per-channel weights that were applied.
            ``health_scores`` (list[float])
                Per-sensor health scores used for weighting.
            ``degraded_sensors`` (list[int])
                Zero-based indices of sensors below
                ``degraded_threshold``.
            ``n_degraded`` (int)
                Number of degraded sensors.
            ``latency_ms`` (float)
                End-to-end wall-clock time in milliseconds.

        Raises
        ------
        TypeError
            If ``sequence`` is not a ``numpy.ndarray``.
        ValueError
            On shape mismatch or out-of-range values.
        RuntimeError
            If the LSTM raises an error during the forward pass.
        """
        t_start = time.perf_counter()

        sequence = self._validate_sequence(sequence)
        resolved_scores = self._resolve_health_scores(sequence, health_scores)

        weights = self.compute_channel_weights(resolved_scores)
        weighted_sequence = self.apply_weights(sequence, weights)

        try:
            batch = weighted_sequence[np.newaxis, :, :].astype(np.float32)
            raw_pred = self.lstm_model.predict(batch, verbose=0)
            rul = float(np.clip(raw_pred.flatten()[0], 0.0, self.rul_clip_max))
        except Exception as exc:
            logger.error("LSTM forward pass failed: %s", exc)
            raise RuntimeError(
                f"LSTM prediction failed: {exc}"
            ) from exc

        health_status = self._rul_to_health_status(rul)
        degraded = sorted(
            i
            for i, s in enumerate(resolved_scores.tolist())
            if s < degraded_threshold
        )
        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

        logger.info(
            "RobustInference | rul=%.1f cycles | health=%s | "
            "degraded_sensors=%s | latency=%.1f ms",
            rul,
            health_status,
            degraded,
            latency_ms,
        )

        return {
            "rul_cycles": rul,
            "rul_hours": round(rul * 0.5, 2),
            "health_status": health_status,
            "channel_weights": weights.tolist(),
            "health_scores": resolved_scores.tolist(),
            "degraded_sensors": degraded,
            "n_degraded": len(degraded),
            "latency_ms": latency_ms,
        }

    # ------------------------------------------------------------------
    # Weighting helpers
    # ------------------------------------------------------------------

    def compute_channel_weights(self, health_scores: np.ndarray) -> np.ndarray:
        """
        Convert health scores to normalised channel weights.

        Applies a minimum floor of :attr:`min_weight`, then normalises so
        that the mean weight equals 1.0, preserving overall input magnitude.

        Parameters
        ----------
        health_scores:
            Shape ``(n_sensors,)``, values in ``[0.0, 1.0]``.

        Returns
        -------
        np.ndarray
            Normalised weights, shape ``(n_sensors,)``.

        Raises
        ------
        ValueError
            If ``health_scores`` has the wrong shape.
        """
        if health_scores.shape != (self.n_sensors,):
            raise ValueError(
                f"health_scores must have shape ({self.n_sensors},), "
                f"got {health_scores.shape}"
            )
        raw = np.clip(health_scores, self.min_weight, 1.0)
        # Normalise: mean weight = 1.0 so the LSTM's expected input scale
        # is preserved even when some channels are heavily down-weighted.
        total = raw.sum()
        if total < 1e-12:
            # Degenerate case: every sensor floored at min_weight.
            return np.full(self.n_sensors, 1.0)
        return raw * (self.n_sensors / total)

    def apply_weights(
        self, sequence: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """
        Multiply each sensor channel by its corresponding weight.

        Parameters
        ----------
        sequence:
            Shape ``(seq_len, n_sensors)``.
        weights:
            Shape ``(n_sensors,)``.

        Returns
        -------
        np.ndarray
            Weighted sequence; shape unchanged.

        Raises
        ------
        ValueError
            If ``weights`` has the wrong shape.
        """
        if weights.shape != (self.n_sensors,):
            raise ValueError(
                f"weights must have shape ({self.n_sensors},), "
                f"got {weights.shape}"
            )
        # Broadcast weights across the time dimension.
        return sequence * weights[np.newaxis, :]

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_model_manager(
        cls,
        model_manager: Any,
        health_monitor: Optional[SensorHealthMonitor] = None,
        **kwargs: Any,
    ) -> "RobustInferenceEngine":
        """
        Build a :class:`RobustInferenceEngine` from the global ModelManager.

        This is the recommended way to wire the engine into the FastAPI
        application startup, alongside the existing ``InferenceEngine``::

            from ..models.model_manager import model_manager
            from robustness.robust_inference import RobustInferenceEngine

            robust_engine = RobustInferenceEngine.from_model_manager(
                model_manager, health_monitor=monitor
            )

        Parameters
        ----------
        model_manager:
            The application-level ``ModelManager`` singleton
            (``from ..models.model_manager import model_manager``).
        health_monitor:
            Optional pre-built :class:`~robustness.sensor_health_monitor.SensorHealthMonitor`.
        **kwargs:
            Forwarded to :class:`RobustInferenceEngine.__init__`.

        Returns
        -------
        RobustInferenceEngine

        Raises
        ------
        RuntimeError
            If the LSTM model has not been loaded into ``model_manager``.
        """
        lstm = model_manager.get_model("lstm")
        if lstm is None:
            raise RuntimeError(
                "LSTM model is not loaded in ModelManager. "
                "Ensure the service has started and models are initialised."
            )
        return cls(lstm_model=lstm, health_monitor=health_monitor, **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_sequence(self, sequence: np.ndarray) -> np.ndarray:
        if not isinstance(sequence, np.ndarray):
            raise TypeError(
                f"sequence must be a numpy.ndarray, got {type(sequence).__name__}"
            )
        sequence = sequence.astype(np.float32, copy=False)
        if sequence.ndim != 2:
            raise ValueError(
                f"sequence must be 2-D (seq_len, n_sensors), "
                f"got shape {sequence.shape}"
            )
        if sequence.shape[1] != self.n_sensors:
            raise ValueError(
                f"sequence has {sequence.shape[1]} sensor columns; "
                f"expected {self.n_sensors}"
            )
        if not np.isfinite(sequence).all():
            raise ValueError("sequence contains NaN or Inf values")
        return sequence

    def _resolve_health_scores(
        self,
        sequence: np.ndarray,
        health_scores: Optional[np.ndarray],
    ) -> np.ndarray:
        """Return health scores from argument, monitor, or uniform fallback."""
        if health_scores is not None:
            hs = np.asarray(health_scores, dtype=np.float64)
            if hs.shape != (self.n_sensors,):
                raise ValueError(
                    f"health_scores must have shape ({self.n_sensors},), "
                    f"got {hs.shape}"
                )
            if not np.all((hs >= 0.0) & (hs <= 1.0)):
                raise ValueError(
                    "All health_scores must be in [0.0, 1.0]"
                )
            return hs

        if self.health_monitor is not None:
            try:
                return self.health_monitor.compute_health_scores(sequence)
            except Exception as exc:
                logger.warning(
                    "Health monitor failed (%s) — falling back to uniform scores",
                    exc,
                )

        logger.debug(
            "No health scores provided — applying uniform channel weights"
        )
        return np.ones(self.n_sensors, dtype=np.float64)

    @staticmethod
    def _rul_to_health_status(rul: float) -> str:
        """Map a RUL value (cycles) to a health status string."""
        if rul >= 100:
            return "healthy"
        if rul >= 50:
            return "warning"
        if rul >= 10:
            return "critical"
        return "imminent_failure"


# ---------------------------------------------------------------------------
# FastAPI router (optional — only registered when FastAPI is available)
# ---------------------------------------------------------------------------

if _FASTAPI_AVAILABLE:

    class _SensorReading(BaseModel):
        """A single time-step of raw sensor readings (21 C-MAPSS channels)."""

        values: List[float] = Field(
            ...,
            description=(
                "21 normalised C-MAPSS sensor values in channel order: "
                "T2, T24, T30, T50, P2, P15, P30, Nf, Nc, epr, Ps30, phi, "
                "NRf, NRc, BPR, farB, htBleed, Nf_dmd, PCNfR_dmd, W31, W32"
            ),
            min_length=21,
            max_length=21,
        )

    class RobustRULRequest(BaseModel):
        """Request body for the ``POST /predict/robust_rul`` endpoint."""

        equipment_id: str = Field(..., description="Equipment identifier")
        sequence: List[_SensorReading] = Field(
            ...,
            description="Time-ordered sensor readings (≥1, ≤200 time steps)",
            min_length=1,
            max_length=200,
        )
        health_scores: Optional[List[float]] = Field(
            None,
            description=(
                "Pre-computed per-sensor health scores in [0, 1], "
                "shape (21,).  When omitted, computed automatically "
                "if a SensorHealthMonitor is attached to the engine."
            ),
            min_length=21,
            max_length=21,
        )
        degraded_threshold: float = Field(
            0.5,
            ge=0.0,
            le=1.0,
            description="Sensors below this score are reported as degraded.",
        )

        class Config:
            json_schema_extra = {
                "example": {
                    "equipment_id": "ENGINE_0042",
                    "sequence": [
                        {"values": [518.67, 641.82, 1589.70, 1400.60, 14.62,
                                    21.61, 554.36, 2388.06, 9046.19, 1.30,
                                    47.47, 521.66, 2388.06, 8138.62, 8.4195,
                                    0.03, 392, 2388.0, 100.0, 39.06, 23.4190]}
                    ],
                    "health_scores": None,
                    "degraded_threshold": 0.5,
                }
            }

    class RobustRULResponse(BaseModel):
        """Response from the ``POST /predict/robust_rul`` endpoint."""

        equipment_id: str
        rul_cycles: float = Field(..., description="Predicted RUL (engine cycles)")
        rul_hours: float = Field(
            ..., description="Approximate RUL in hours (1 cycle ≈ 0.5 h)"
        )
        health_status: str = Field(
            ...,
            description=(
                "Derived health status: healthy | warning | critical | "
                "imminent_failure"
            ),
        )
        channel_weights: List[float] = Field(
            ..., description="Normalised per-sensor weights applied before LSTM"
        )
        health_scores: List[float] = Field(
            ..., description="Per-sensor health scores (0=degraded, 1=healthy)"
        )
        degraded_sensors: List[int] = Field(
            ..., description="Zero-based indices of sensors below the threshold"
        )
        n_degraded: int = Field(..., description="Number of degraded sensors")
        latency_ms: float = Field(..., description="End-to-end inference latency (ms)")

    # Module-level engine reference set by the FastAPI app at startup.
    # Attach via: ``import robustness.robust_inference as ri``
    #             ``ri._engine_ref = RobustInferenceEngine.from_model_manager(mm)``
    _engine_ref: Optional[RobustInferenceEngine] = None

    robust_router = APIRouter()

    @robust_router.post(
        "/robust_rul",
        response_model=RobustRULResponse,
        summary="Robust RUL prediction with sensor-health weighting",
        description=(
            "Predicts Remaining Useful Life using the LSTM model with "
            "dynamic down-weighting of degraded sensor channels.  "
            "Sensor health scores are computed automatically via the "
            "autoencoder-based SensorHealthMonitor when not supplied."
        ),
    )
    async def predict_robust_rul(request: RobustRULRequest) -> RobustRULResponse:
        """
        Robust RUL endpoint — wraps :meth:`RobustInferenceEngine.predict_rul`.
        """
        if _engine_ref is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "RobustInferenceEngine not initialised. "
                    "Ensure robustness.robust_inference._engine_ref is set "
                    "during application startup."
                ),
            )

        # Convert the list-of-readings request into a numpy array.
        try:
            seq_array = np.array(
                [step.values for step in request.sequence], dtype=np.float32
            )  # (seq_len, 21)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to parse sensor sequence: {exc}",
            )

        hs_array: Optional[np.ndarray] = None
        if request.health_scores is not None:
            hs_array = np.array(request.health_scores, dtype=np.float64)
            if not np.all((hs_array >= 0.0) & (hs_array <= 1.0)):
                raise HTTPException(
                    status_code=422,
                    detail="All health_scores must be in [0.0, 1.0]",
                )

        try:
            result = _engine_ref.predict_rul(
                sequence=seq_array,
                health_scores=hs_array,
                degraded_threshold=request.degraded_threshold,
            )
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except RuntimeError as exc:
            logger.error("Robust inference error: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))

        return RobustRULResponse(
            equipment_id=request.equipment_id,
            **result,
        )
