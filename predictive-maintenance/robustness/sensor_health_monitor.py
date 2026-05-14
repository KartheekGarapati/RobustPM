"""
Per-sensor health scoring using autoencoder reconstruction error.

Assigns each of the 21 NASA C-MAPSS sensor channels a health score in
**[0.0, 1.0]**, where 1.0 means fully healthy and 0.0 means fully
degraded.  The score is derived from how far each sensor's actual reading
deviates from what the autoencoder expects for a healthy engine.

Architecture overview
---------------------
A symmetric dense autoencoder is trained on *healthy* sensor data
(early-life or low-RUL-degradation samples).  At inference time the
per-sensor squared reconstruction error is compared against per-sensor
95th-percentile thresholds established during training::

    raw_error[i]  = (reading[i] − reconstruction[i]) ** 2
    health[i]     = 1 − clip(raw_error[i] / threshold[i], 0, 1)

Typical usage::

    monitor = SensorHealthMonitor(n_sensors=21, encoding_dim=8)
    monitor.fit(healthy_data)                  # (n_samples, 21)
    scores = monitor.compute_health_scores(window)  # (21,) in [0, 1]
    monitor.save("models/sensor_monitor")

    # Load in a new process:
    monitor2 = SensorHealthMonitor()
    monitor2.load("models/sensor_monitor")
    scores = monitor2.compute_health_scores(reading)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    _TF_AVAILABLE = True
except (ImportError, ValueError):
    # ValueError is raised when h5py has a binary incompatibility with numpy
    # (e.g. h5py compiled against a different numpy ABI).
    tf = None  # type: ignore[assignment]
    keras = None  # type: ignore[assignment]
    layers = None  # type: ignore[assignment]
    _TF_AVAILABLE = False
    logger.warning(
        "TensorFlow not available — SensorHealthMonitor requires TensorFlow"
    )


class SensorHealthMonitor:
    """
    Monitor per-sensor health via autoencoder reconstruction error.

    Parameters
    ----------
    n_sensors:
        Number of raw sensor channels.  21 for the NASA C-MAPSS dataset.
    encoding_dim:
        Bottleneck dimension of the autoencoder.  Smaller values force
        stronger compression and increase anomaly sensitivity.
    model_path:
        Optional path to a directory previously saved with :meth:`save`.
        When provided the autoencoder and calibration thresholds are
        loaded immediately; ``fit()`` is not required.

    Raises
    ------
    ImportError
        If TensorFlow is not installed.
    FileNotFoundError
        If ``model_path`` is provided but does not exist.
    """

    N_SENSORS: int = 21
    _AUTOENCODER_SUBDIR: str = "autoencoder"
    _THRESHOLD_FILE: str = "calibration_thresholds.npy"

    def __init__(
        self,
        n_sensors: int = 21,
        encoding_dim: int = 8,
        model_path: Optional[str] = None,
    ) -> None:
        if not _TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is required for SensorHealthMonitor. "
                "Install it with: pip install tensorflow"
            )

        if n_sensors < 1:
            raise ValueError(f"n_sensors must be >= 1, got {n_sensors}")
        if encoding_dim < 1:
            raise ValueError(f"encoding_dim must be >= 1, got {encoding_dim}")

        self.n_sensors = n_sensors
        self.encoding_dim = encoding_dim
        self.autoencoder: Optional[tf.keras.Model] = None
        # Per-sensor upper-bound errors set during fit() / load().
        # Shape: (n_sensors,).
        self._calibration_thresholds: Optional[np.ndarray] = None

        if model_path is not None:
            self.load(model_path)
        else:
            self.autoencoder = self._build_autoencoder()

        logger.info(
            "SensorHealthMonitor initialised | n_sensors=%d | encoding_dim=%d",
            self.n_sensors,
            self.encoding_dim,
        )

    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------

    def _build_autoencoder(self) -> "tf.keras.Model":
        """
        Build and compile a symmetric dense autoencoder.

        The encoder compresses ``n_sensors`` → 64 → 32 → ``encoding_dim``
        with ReLU activations and batch normalisation.  The decoder mirrors
        this path back to ``n_sensors`` with a linear output (no activation),
        which is appropriate for normalised sensor data.

        Returns
        -------
        tf.keras.Model
            Compiled autoencoder ready for :meth:`fit`.
        """
        inputs = keras.Input(shape=(self.n_sensors,), name="sensor_input")

        # Encoder
        x = layers.Dense(64, activation="relu", name="enc_dense_1")(inputs)
        x = layers.BatchNormalization(name="enc_bn_1")(x)
        x = layers.Dense(32, activation="relu", name="enc_dense_2")(x)
        encoded = layers.Dense(
            self.encoding_dim, activation="relu", name="bottleneck"
        )(x)

        # Decoder
        x = layers.Dense(32, activation="relu", name="dec_dense_1")(encoded)
        x = layers.Dense(64, activation="relu", name="dec_dense_2")(x)
        reconstructed = layers.Dense(
            self.n_sensors, activation="linear", name="reconstruction"
        )(x)

        model = keras.Model(
            inputs=inputs,
            outputs=reconstructed,
            name="sensor_autoencoder",
        )
        model.compile(optimizer="adam", loss="mse")

        logger.debug(
            "Autoencoder built | parameters=%d", model.count_params()
        )
        return model

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        epochs: int = 50,
        batch_size: int = 64,
        validation_split: float = 0.1,
        calibration_percentile: float = 95.0,
    ) -> "tf.keras.callbacks.History":
        """
        Train the autoencoder on healthy sensor data and calibrate thresholds.

        After training, per-sensor calibration thresholds are derived from
        the ``calibration_percentile`` of the per-sensor squared
        reconstruction errors on the *training* set.  These thresholds
        define what "maximum healthy error" looks like for each sensor.

        Parameters
        ----------
        X_train:
            Healthy sensor data of shape ``(n_samples, n_sensors)``.
            Values should be z-score or min-max normalised to stabilise
            training.
        epochs:
            Maximum training epochs (early stopping typically fires first).
        batch_size:
            Mini-batch size.
        validation_split:
            Fraction of training samples reserved for monitoring.
        calibration_percentile:
            Percentile of per-sensor training error used as the 1.0 → 0.0
            health boundary.  95 is recommended; lower values make the
            monitor more sensitive.

        Returns
        -------
        tf.keras.callbacks.History
            Keras History object from the underlying ``model.fit()`` call.

        Raises
        ------
        ValueError
            If ``X_train`` has an incompatible shape or contains NaN/Inf.
        """
        self._validate_input_array(X_train, context="X_train")

        if not 0.0 < calibration_percentile <= 100.0:
            raise ValueError(
                f"calibration_percentile must be in (0, 100], "
                f"got {calibration_percentile}"
            )

        logger.info(
            "Training autoencoder | samples=%d | epochs=%d | batch=%d",
            len(X_train),
            epochs,
            batch_size,
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                verbose=0,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=0,
            ),
        ]

        history = self.autoencoder.fit(
            X_train,
            X_train,  # reconstruction target = input
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1,
        )

        # Calibrate per-sensor thresholds from training errors.
        train_errors = self._per_sensor_squared_error(X_train)
        self._calibration_thresholds = np.percentile(
            train_errors, calibration_percentile, axis=0
        )
        # Replace zero thresholds (constant sensors) with a small epsilon.
        self._calibration_thresholds = np.where(
            self._calibration_thresholds < 1e-10,
            1e-10,
            self._calibration_thresholds,
        )

        logger.info(
            "Calibration complete | percentile=%g | mean_threshold=%.6f | "
            "max_threshold=%.6f",
            calibration_percentile,
            float(self._calibration_thresholds.mean()),
            float(self._calibration_thresholds.max()),
        )
        return history

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def compute_health_scores(self, sensor_reading: np.ndarray) -> np.ndarray:
        """
        Compute per-sensor health scores for a sensor reading or window.

        When a 2-D array (time-window) is supplied, the per-sensor mean
        reconstruction error over the window is used, which smooths
        transient measurement noise.

        Parameters
        ----------
        sensor_reading:
            Shape ``(n_sensors,)`` for a single reading or
            ``(n_timesteps, n_sensors)`` for a temporal window.

        Returns
        -------
        np.ndarray
            Health scores, shape ``(n_sensors,)``, values in ``[0.0, 1.0]``.
            **1.0** = healthy, **0.0** = fully degraded.

        Raises
        ------
        RuntimeError
            If the monitor has not been calibrated (``fit()`` not called and
            no pre-trained model loaded).
        ValueError
            On shape mismatch or NaN/Inf values.
        """
        if self._calibration_thresholds is None:
            raise RuntimeError(
                "SensorHealthMonitor is not calibrated. "
                "Call fit() or load a pre-trained model first."
            )

        self._validate_input_array(sensor_reading, context="sensor_reading")

        # Normalise to 2-D for uniform processing.
        X = (
            sensor_reading[np.newaxis, :]
            if sensor_reading.ndim == 1
            else sensor_reading
        )

        # Per-sample, per-sensor squared error → mean over samples.
        per_sample_errors = self._per_sensor_squared_error(X)  # (n, n_sensors)
        per_sensor_errors = per_sample_errors.mean(axis=0)     # (n_sensors,)

        # Normalise by calibration thresholds and invert to health score.
        normalised = np.clip(
            per_sensor_errors / self._calibration_thresholds, 0.0, 1.0
        )
        health_scores = 1.0 - normalised

        logger.debug(
            "Health scores computed | min=%.3f | mean=%.3f | max=%.3f",
            float(health_scores.min()),
            float(health_scores.mean()),
            float(health_scores.max()),
        )
        return health_scores

    def identify_degraded_sensors(
        self,
        health_scores: np.ndarray,
        threshold: float = 0.5,
    ) -> list:
        """
        Return indices of sensors whose health score is below ``threshold``.

        Parameters
        ----------
        health_scores:
            Shape ``(n_sensors,)`` as returned by :meth:`compute_health_scores`.
        threshold:
            Sensors with scores strictly below this value are flagged.
            Default ``0.5`` (half-degraded).

        Returns
        -------
        list[int]
            Sorted list of zero-based sensor indices considered degraded.

        Raises
        ------
        ValueError
            If ``threshold`` is outside ``[0, 1]``.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"threshold must be in [0.0, 1.0], got {threshold}"
            )
        degraded = [
            i
            for i, score in enumerate(health_scores.tolist())
            if score < threshold
        ]
        if degraded:
            logger.warning(
                "%d sensor(s) degraded (score < %.2f): indices %s",
                len(degraded),
                threshold,
                degraded,
            )
        return sorted(degraded)

    def get_reconstruction_error(self, sensor_reading: np.ndarray) -> np.ndarray:
        """
        Return the raw per-sensor squared reconstruction error.

        Useful for diagnostics and threshold tuning.

        Parameters
        ----------
        sensor_reading:
            Shape ``(n_sensors,)`` or ``(n_timesteps, n_sensors)``.

        Returns
        -------
        np.ndarray
            Mean per-sensor squared error, shape ``(n_sensors,)``.
        """
        self._validate_input_array(sensor_reading, context="sensor_reading")
        X = (
            sensor_reading[np.newaxis, :]
            if sensor_reading.ndim == 1
            else sensor_reading
        )
        return self._per_sensor_squared_error(X).mean(axis=0)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Persist the autoencoder weights and calibration thresholds to disk.

        Creates ``<path>/autoencoder/`` (Keras SavedModel format) and
        ``<path>/calibration_thresholds.npy``.

        Parameters
        ----------
        path:
            Target directory.  Created if it does not exist.

        Raises
        ------
        RuntimeError
            If ``autoencoder`` has not been built.
        """
        if self.autoencoder is None:
            raise RuntimeError("No autoencoder to save")

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        self.autoencoder.save(str(save_dir / self._AUTOENCODER_SUBDIR))

        if self._calibration_thresholds is not None:
            np.save(
                str(save_dir / self._THRESHOLD_FILE),
                self._calibration_thresholds,
            )
        else:
            logger.warning(
                "Saving without calibration thresholds — call fit() first"
            )

        logger.info("SensorHealthMonitor saved to %s", save_dir)

    def load(self, path: str) -> None:
        """
        Load a previously saved autoencoder and calibration thresholds.

        Parameters
        ----------
        path:
            Directory previously passed to :meth:`save`.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        """
        load_dir = Path(path)
        if not load_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {load_dir}"
            )

        ae_dir = load_dir / self._AUTOENCODER_SUBDIR
        if not ae_dir.exists():
            raise FileNotFoundError(
                f"Autoencoder sub-directory not found: {ae_dir}"
            )

        self.autoencoder = keras.models.load_model(str(ae_dir))

        thresh_file = load_dir / self._THRESHOLD_FILE
        if thresh_file.exists():
            self._calibration_thresholds = np.load(str(thresh_file))
        else:
            logger.warning(
                "No calibration thresholds found at %s — "
                "compute_health_scores() will raise until fit() is called",
                thresh_file,
            )

        logger.info("SensorHealthMonitor loaded from %s", load_dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _per_sensor_squared_error(self, X: np.ndarray) -> np.ndarray:
        """
        Return per-sample, per-sensor squared reconstruction error.

        Parameters
        ----------
        X:
            Shape ``(n_samples, n_sensors)``.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples, n_sensors)``.
        """
        reconstructed = self.autoencoder.predict(X, verbose=0)
        return (X - reconstructed) ** 2

    def _validate_input_array(
        self, X: np.ndarray, context: str = "input"
    ) -> None:
        """Raise informative errors for shape or value problems."""
        if not isinstance(X, np.ndarray):
            raise TypeError(
                f"{context} must be a numpy.ndarray, got {type(X).__name__}"
            )
        if X.ndim == 1:
            if X.shape[0] != self.n_sensors:
                raise ValueError(
                    f"{context} has {X.shape[0]} elements; "
                    f"expected {self.n_sensors} (n_sensors)"
                )
        elif X.ndim == 2:
            if X.shape[1] != self.n_sensors:
                raise ValueError(
                    f"{context} has {X.shape[1]} columns; "
                    f"expected {self.n_sensors} (n_sensors)"
                )
        else:
            raise ValueError(
                f"{context} must be 1-D or 2-D, got {X.ndim}-D "
                f"with shape {X.shape}"
            )
        if not np.isfinite(X).all():
            raise ValueError(
                f"{context} contains NaN or Inf values"
            )
