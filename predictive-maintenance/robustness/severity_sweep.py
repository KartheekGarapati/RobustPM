#!/usr/bin/env python3
"""
severity_sweep.py — Robustness layer evaluation across fault severity 0.0-1.0.

Fixes Engine 1 from FD001 (cycles 82-111, true RUL=81) and sweeps fault
severity from 0.0 to 1.0 in steps of 0.1 for all 4 fault modes, recording
the degraded and robust RUL error at each level.

Generates severity_sweep.png: 2×2 figure with one line-plot per fault mode
showing how |RUL error| evolves with severity and the recovery gap between
the degraded (red) and robust (green) predictions.

Usage
-----
From the project root:
    python predictive-maintenance/robustness/severity_sweep.py

From predictive-maintenance/:
    python robustness/severity_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent       # .../robustness/
_PM_DIR     = _SCRIPT_DIR.parent                    # .../predictive-maintenance/
_ROOT       = _PM_DIR.parent                        # project root

sys.path.insert(0, str(_PM_DIR))
from robustness.sensor_degradation import SensorDegradationInjector, CMAPSS_SENSOR_NAMES

# ── configuration ─────────────────────────────────────────────────────────────
TEST_FILE  = _ROOT / "archive" / "CMaps" / "test_FD001.txt"
RUL_FILE   = _ROOT / "archive" / "CMaps" / "RUL_FD001.txt"
PNG_OUT    = _SCRIPT_DIR / "severity_sweep.png"

ENGINE_ID  = 1
WINDOW     = 30            # last N cycles  →  cycles 82-111 for engine 1
AFFECTED   = [2, 5, 10]   # T30, P15, Ps30
SEED_BASE  = 42

SEVERITIES: list[float] = [round(s * 0.1, 1) for s in range(11)]   # 0.0 … 1.0
MODES: list[str] = [
    "gaussian_noise",
    "stuck_at_value",
    "partial_dropout",
    "linear_drift",
]
MODE_LABELS: Dict[str, str] = {
    "gaussian_noise":  "Gaussian Noise",
    "stuck_at_value":  "Stuck at Value",
    "partial_dropout": "Partial Dropout",
    "linear_drift":    "Linear Drift",
}

N_SENSORS  = 21
MIN_WEIGHT = 0.1
BIAS_SCALE = 80.0
N_NOISE    = 4.0
W          = 70


# ── helpers ───────────────────────────────────────────────────────────────────

def sep(char: str = "-", width: int = W) -> None:
    print(char * width)


# ── data loading ──────────────────────────────────────────────────────────────

def load_test_engines(path: Path) -> Dict[int, np.ndarray]:
    raw   = np.loadtxt(str(path), dtype=np.float64)
    units = raw[:, 0].astype(int)
    engines: Dict[int, np.ndarray] = {}
    for uid in np.unique(units):
        mask = units == uid
        engines[int(uid)] = raw[mask, 5:26].astype(np.float64)
    return engines


def load_true_rul(path: Path) -> Dict[int, float]:
    rul_vals = np.loadtxt(str(path), dtype=np.float64)
    return {int(i + 1): float(v) for i, v in enumerate(rul_vals)}


# ── numpy-only health scoring (mirrors SensorHealthMonitor) ──────────────────

def compute_health_scores(clean: np.ndarray, noisy: np.ndarray) -> np.ndarray:
    err = ((noisy - clean) ** 2).mean(axis=0)
    thr = np.maximum(np.var(clean, axis=0), 1e-10)
    return 1.0 - np.clip(err / thr, 0.0, 1.0)


def compute_weights(health: np.ndarray) -> np.ndarray:
    raw = np.maximum(health, MIN_WEIGHT)
    return raw * (N_SENSORS / raw.sum())


# ── mock LSTM (same physics-motivated estimator as benchmark.py) ──────────────

def mock_lstm(
    data: np.ndarray,
    reference: np.ndarray,
    true_rul: float,
    rng: np.random.Generator,
) -> float:
    dev  = float(np.sqrt(np.mean((data - reference) ** 2)))
    bias = dev * BIAS_SCALE
    return float(np.clip(true_rul - bias + rng.normal(0.0, N_NOISE), 1.0, 350.0))


# ── sweep ─────────────────────────────────────────────────────────────────────

# results[mode][severity] = {"degraded_error": float, "robust_error": float}
SweepResults = Dict[str, Dict[float, Dict[str, float]]]


def run_sweep(clean: np.ndarray, true_rul: float) -> tuple[SweepResults, float]:
    """
    Sweep all 4 fault modes over severities 0.0–1.0 on a fixed clean window.

    Returns (results, clean_baseline_error).
    """
    # Clean baseline — no injection, deterministic seed
    clean_rul = mock_lstm(clean, clean, true_rul, np.random.default_rng(SEED_BASE))
    clean_err = clean_rul - true_rul

    results: SweepResults = {}

    for m_idx, mode in enumerate(MODES):
        results[mode] = {}
        for s_idx, sev in enumerate(SEVERITIES):
            # Deterministic seeds per (mode, severity) combination
            inj_seed = SEED_BASE + 1000 * (m_idx + 1) + s_idx
            deg_seed = SEED_BASE + 2000 * (m_idx + 1) + s_idx
            rob_seed = SEED_BASE + 3000 * (m_idx + 1) + s_idx

            injector = SensorDegradationInjector(
                fault_severity=sev,
                affected_sensor_indices=AFFECTED,
                random_seed=inj_seed,
            )
            noisy = injector.inject(clean, mode)

            # Degraded prediction
            deg_rul = mock_lstm(noisy, clean, true_rul, np.random.default_rng(deg_seed))

            # Robust prediction: health-weighted channels
            health  = compute_health_scores(clean, noisy)
            weights = compute_weights(health)
            w_noisy = noisy * weights[np.newaxis, :]
            w_clean = clean * weights[np.newaxis, :]
            rob_rul = mock_lstm(w_noisy, w_clean, true_rul, np.random.default_rng(rob_seed))

            results[mode][sev] = {
                "degraded_error": deg_rul - true_rul,
                "robust_error":   rob_rul - true_rul,
            }

    return results, clean_err


# ── plotting ──────────────────────────────────────────────────────────────────

def generate_plot(
    results: SweepResults,
    clean_err: float,
    path: Path,
    true_rul: float,
    cycle_start: int,
    cycle_end: int,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot] matplotlib not available — skipping PNG generation")
        return

    baseline_abs = abs(clean_err)
    sevs = SEVERITIES

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"RobustPM — Severity Sweep: Engine {ENGINE_ID}, FD001  "
        f"(cycles {cycle_start}–{cycle_end}, true RUL = {true_rul:.0f})\n"
        f"Affected sensors: {[CMAPSS_SENSOR_NAMES[i] for i in AFFECTED]}",
        fontsize=13,
        fontweight="bold",
    )

    for ax, mode in zip(axes.flat, MODES):
        mode_data = results[mode]
        deg_abs = [abs(mode_data[s]["degraded_error"]) for s in sevs]
        rob_abs = [abs(mode_data[s]["robust_error"])   for s in sevs]

        # Shaded recovery gap between the two curves
        ax.fill_between(
            sevs, deg_abs, rob_abs,
            alpha=0.20,
            color="mediumseagreen",
            label="Recovery gap",
        )

        # Main lines
        ax.plot(sevs, deg_abs, color="tomato",         linewidth=2.2,
                marker="o", markersize=5, label="Degraded RUL error")
        ax.plot(sevs, rob_abs, color="mediumseagreen",  linewidth=2.2,
                marker="s", markersize=5, label="Robust RUL error")
        ax.axhline(
            baseline_abs,
            color="royalblue", linewidth=1.8, linestyle="--",
            label=f"Clean baseline ({baseline_abs:.1f} cyc)",
        )

        # Annotate the worst-severity recovery
        max_gain = max(d - r for d, r in zip(deg_abs, rob_abs))
        ax.text(
            0.97, 0.97,
            f"Peak recovery: {max_gain:.1f} cyc",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.85),
        )

        ax.set_title(MODE_LABELS[mode], fontsize=12, fontweight="bold")
        ax.set_xlabel("Fault Severity", fontsize=10)
        ax.set_ylabel("|RUL Error| (cycles)", fontsize=10)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(bottom=0)
        ax.set_xticks(sevs)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(linestyle="--", alpha=0.4)
        ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved -> {path}")


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary(results: SweepResults, clean_err: float, true_rul: float) -> None:
    baseline_abs = abs(clean_err)
    sep("=")
    print(f"  RobustPM — Severity Sweep Summary  |  Engine {ENGINE_ID}, FD001  "
          f"|  True RUL = {true_rul:.0f}")
    sep("=")
    print(f"  Clean baseline |error| : {baseline_abs:.2f} cycles")
    print(f"  Affected sensors       : "
          f"{[CMAPSS_SENSOR_NAMES[i] for i in AFFECTED]}  (indices {AFFECTED})")
    sep("-")
    print(
        f"  {'Mode':<22}{'Sev':>5}"
        f"{'|Deg Err|':>12}{'|Rob Err|':>12}{'Recovery %':>13}"
    )
    sep("-")

    for mode in MODES:
        for sev in SEVERITIES:
            d       = results[mode][sev]
            deg_abs = abs(d["degraded_error"])
            rob_abs = abs(d["robust_error"])
            rec = (
                (deg_abs - rob_abs) / deg_abs * 100.0
                if deg_abs > 1e-6 else 100.0
            )
            flag = " <" if rob_abs > deg_abs else ""    # mark when robust is worse
            print(
                f"  {mode.replace('_', ' '):<22}{sev:>5.1f}"
                f"{deg_abs:>12.2f}{rob_abs:>12.2f}"
                f"{rec:>12.1f}%{flag}"
            )
        sep("-")

    sep("=")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    sep("=")
    print("  RobustPM — Fault Severity Sweep  (Engine 1, FD001)")
    sep("=")
    print(f"  Test data  : {TEST_FILE}")
    print(f"  RUL data   : {RUL_FILE}")
    print(f"  PNG output : {PNG_OUT}")
    print(f"  Severities : {SEVERITIES}")
    sep("-")

    print("\n  Loading test data...")
    engines   = load_test_engines(TEST_FILE)
    true_ruls = load_true_rul(RUL_FILE)

    sensors_full = engines[ENGINE_ID]
    true_rul     = true_ruls[ENGINE_ID]
    n_cycles     = sensors_full.shape[0]

    print(f"  Engine {ENGINE_ID}: {n_cycles} total cycles | true RUL = {true_rul:.0f}")
    window_start = n_cycles - WINDOW + 1
    print(f"  Using cycles {window_start}–{n_cycles}  (last {WINDOW} cycles)")

    # Z-score normalize the window (same as benchmark.py)
    window  = sensors_full[max(0, n_cycles - WINDOW):]
    ch_mean = window.mean(axis=0)
    ch_std  = np.where(window.std(axis=0) < 1e-8, 1.0, window.std(axis=0))
    clean   = (window - ch_mean) / ch_std

    n_evals = len(SEVERITIES) * len(MODES)
    print(f"\n  Running {len(SEVERITIES)} severities × {len(MODES)} modes "
          f"= {n_evals} evaluations...")
    sep("-")

    results, clean_err = run_sweep(clean, true_rul)

    print(f"  Sweep complete.")
    print(f"  Clean baseline |error| = {abs(clean_err):.2f} cycles")

    print("\n  Generating plot...")
    generate_plot(results, clean_err, PNG_OUT,
                  true_rul=true_rul,
                  cycle_start=window_start,
                  cycle_end=n_cycles)

    print()
    print_summary(results, clean_err, true_rul)


if __name__ == "__main__":
    main()
