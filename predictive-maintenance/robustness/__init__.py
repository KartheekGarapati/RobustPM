"""
robustness — Sensor fault injection, health monitoring, and robust RUL inference.

Modules
-------
sensor_degradation
    SensorDegradationInjector: inject Gaussian noise, stuck-at-value,
    partial dropout, and linear drift into C-MAPSS sensor channels.

sensor_health_monitor
    SensorHealthMonitor: per-sensor health scoring (0–1) derived from
    autoencoder reconstruction error.

robust_inference
    RobustInferenceEngine: LSTM RUL prediction with dynamic channel
    down-weighting based on sensor health scores.
"""

from .sensor_degradation import SensorDegradationInjector

# TensorFlow-dependent modules are imported lazily so that the package remains
# usable (e.g. for fault injection only) even when TF is not installed or has
# a binary incompatibility with h5py.
try:
    from .sensor_health_monitor import SensorHealthMonitor
    from .robust_inference import RobustInferenceEngine
    __all__ = [
        "SensorDegradationInjector",
        "SensorHealthMonitor",
        "RobustInferenceEngine",
    ]
except (ImportError, ValueError):
    __all__ = ["SensorDegradationInjector"]
