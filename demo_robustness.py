#!/usr/bin/env python3
"""
demo_robustness.py — Robustness layer end-to-end demonstration.

Loads real NASA C-MAPSS FD001 data, injects all four fault modes on sensors
T2 / P15 / Ps30 at severity=0.50, computes per-sensor health scores, and
compares clean vs. degraded vs. robust RUL predictions.

Only numpy is required; TF / pandas / sklearn are bypassed entirely.
Health scoring and RUL estimation are implemented inline as physics-motivated
numpy approximations of the production SensorHealthMonitor and LSTM.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "predictive-maintenance"))

from robustness.sensor_degradation import SensorDegradationInjector, CMAPSS_SENSOR_NAMES

# ── configuration ─────────────────────────────────────────────────────────────
DATA_FILE  = os.path.join(_HERE, "archive", "CMaps", "train_FD001.txt")
ENGINE_ID  = 1
WINDOW_LEN = 30
AFFECTED   = [0, 5, 10]   # T2, P15, Ps30
SEVERITY   = 0.5
SEED       = 42
MODES      = ["gaussian_noise", "stuck_at_value", "partial_dropout", "linear_drift"]
MIN_WEIGHT = 0.1
BIAS_SCALE = 80.0          # cycles per normalised L2 deviation unit
N_SENSORS  = 21
W          = 74            # line width


# ── print helpers ─────────────────────────────────────────────────────────────

def sep(char: str = "-", width: int = W) -> None:
    print(char * width)


def show_table(data: np.ndarray, title: str, cols: list) -> None:
    """Print first-3 / … / last-3 rows for selected sensor columns."""
    names = [CMAPSS_SENSOR_NAMES[i] for i in cols]
    hdr = f"  {'Cyc':>4}  " + "  ".join(f"{n:>9}" for n in names)
    print(f"\n    {title}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    sentinel = -999
    rows = list(range(3)) + [sentinel] + list(range(len(data) - 3, len(data)))
    for r in rows:
        if r == sentinel:
            print(f"  {'...':>4}")
            continue
        vals = "  ".join(f"{data[r, i]:9.4f}" for i in cols)
        print(f"  {r + 1:>4}  {vals}")


# ── data loading ──────────────────────────────────────────────────────────────

def load_engine(path: str, engine: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse FD001 txt file (space-separated, no header, 26 cols).

    Returns (sensor_array, cycle_array) for one engine.
    Columns: [unit_id, time_cycle, op1, op2, op3, s1..s21]
    """
    raw     = np.loadtxt(path, dtype=np.float64)
    mask    = raw[:, 0].astype(int) == engine
    cycles  = raw[mask, 1].astype(int)
    sensors = raw[mask, 5:26]        # (n_cycles_engine, 21)
    return sensors, cycles


# ── health scoring (numpy-only approximation) ─────────────────────────────────

def compute_health_scores(clean: np.ndarray, noisy: np.ndarray) -> np.ndarray:
    """
    Per-sensor health score in [0, 1] via normalised reconstruction error.

    Approximates the production autoencoder by treating the clean data as the
    reconstruction target:

        error[i]     = MSE(noisy[:, i] − clean[:, i])
        threshold[i] = Var(clean[:, i])         ← typical healthy-data error
        health[i]    = 1 − clip(error / threshold, 0, 1)

    Unaffected sensors: error ≈ 0 → health = 1.0
    Affected sensors:   error > 0 → health < 1.0
    """
    err = ((noisy - clean) ** 2).mean(axis=0)               # (21,)
    thr = np.var(clean, axis=0)
    thr = np.maximum(thr, 1e-10)
    return 1.0 - np.clip(err / thr, 0.0, 1.0)              # (21,)


# ── mock LSTM (numpy-only) ─────────────────────────────────────────────────────

def mock_lstm_rul(
    data: np.ndarray,
    reference: np.ndarray,
    true_rul: float,
    rng: np.random.Generator,
    scale: float = BIAS_SCALE,
) -> float:
    """
    Simulate an LSTM RUL prediction on (possibly corrupted) sensor data.

    In the real system the LSTM was trained on clean z-scored inputs.
    Corruption shifts the feature distribution, introducing a proportional
    prediction error.  We approximate:

        bias = RMSE(data − reference) × scale
        predicted_rul = clip(true_rul − bias + N(0, 4), 1, 350)
    """
    dev  = float(np.sqrt(np.mean((data - reference) ** 2)))
    bias = dev * scale
    noise = float(rng.normal(0, 4))
    return float(np.clip(true_rul - bias + noise, 1.0, 350.0))


# ── robust prediction (numpy-only) ────────────────────────────────────────────

def robust_predict(
    clean: np.ndarray,
    noisy: np.ndarray,
    true_rul: float,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Robust RUL prediction via health-weighted sensor inputs.

    Steps mirror the production RobustInferenceEngine:
    1. Compute per-sensor health scores.
    2. Derive mean-normalised channel weights (floor at MIN_WEIGHT).
    3. Scale both noisy and clean by the same weight vector (preserves the
       relative feature relationship the LSTM relies on).
    4. Pass scaled inputs to the mock LSTM.

    Returns (rul, weights, health_scores).
    """
    hs      = compute_health_scores(clean, noisy)
    raw_w   = np.maximum(hs, MIN_WEIGHT)
    w       = raw_w * N_SENSORS / raw_w.sum()           # mean-normalised
    w_noisy = noisy * w[np.newaxis, :]
    w_clean = clean * w[np.newaxis, :]                  # same scale shift
    rul = mock_lstm_rul(w_noisy, w_clean, true_rul, rng)
    return rul, w, hs


# ── main demo ─────────────────────────────────────────────────────────────────

def main() -> None:
    sep("=")
    print("  NASA C-MAPSS FD001 — Robustness Layer End-to-End Demonstration")
    sep("=")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print(f"\n[1/6]  Loading C-MAPSS FD001  (engine #{ENGINE_ID})")
    sensors_full, cycles_full = load_engine(DATA_FILE, ENGINE_ID)
    max_cycle = int(cycles_full.max())

    mid = max_cycle // 2
    ws  = mid - WINDOW_LEN // 2
    we  = ws + WINDOW_LEN
    true_rul = max_cycle - we

    window_raw = sensors_full[ws:we]                    # (30, 21) raw counts
    ch_mean    = window_raw.mean(axis=0)
    ch_std     = window_raw.std(axis=0)
    ch_std     = np.where(ch_std < 1e-8, 1.0, ch_std)
    clean      = (window_raw - ch_mean) / ch_std       # z-scored (30, 21)

    aff_names = [CMAPSS_SENSOR_NAMES[i] for i in AFFECTED]
    print(f"       Dataset      : {sensors_full.shape[0]} rows for engine #{ENGINE_ID}")
    print(f"       Engine life  : {max_cycle} cycles")
    print(f"       Window       : cycles {cycles_full[ws]}–{cycles_full[we - 1]}"
          f"  (midpoint ≈ {mid})")
    print(f"       True RUL     : {true_rul} cycles")
    print(f"       Fault targets: {aff_names}  (indices {AFFECTED})")

    # ── 2. Inject all 4 fault modes ──────────────────────────────────────────
    print(f"\n[2/6]  Injecting faults  (severity={SEVERITY})")
    injector = SensorDegradationInjector(
        fault_severity=SEVERITY,
        affected_sensor_indices=AFFECTED,
        random_seed=SEED,
    )
    corrupted: dict[str, np.ndarray] = {}
    for mode in MODES:
        corrupted[mode] = injector.inject(clean, mode)
        delta = float(np.abs(corrupted[mode][:, AFFECTED] - clean[:, AFFECTED]).mean())
        print(f"       {mode:<22s}  mean |Δ| on {aff_names} = {delta:.4f} σ")

    # ── 3. Before / after readings ────────────────────────────────────────────
    print(f"\n[3/6]  Sensor readings — clean vs. faulted  (z-scored units)")
    show_table(clean, "CLEAN baseline:", AFFECTED)
    for mode in MODES:
        label = mode.replace("_", " ").title()
        show_table(corrupted[mode], f"{label}:", AFFECTED)

    # ── 4. Health scores ──────────────────────────────────────────────────────
    print(f"\n[4/6]  Per-sensor health scores  (1.00 = healthy, 0.00 = failed)")
    hs_all: dict[str, np.ndarray] = {
        m: compute_health_scores(clean, corrupted[m]) for m in MODES
    }

    cw  = 17
    hdr = f"  {'Sensor':<12}" + "".join(f"{m[:cw]:>{cw}}" for m in MODES)
    print(hdr)
    sep("-", len(hdr) - 2)
    for i, name in enumerate(CMAPSS_SENSOR_NAMES):
        tag = " *" if i in AFFECTED else ""
        row = f"  {name:<12}" + "".join(f"{hs_all[m][i]:>{cw}.4f}" for m in MODES)
        print(row + tag)
    print("\n  * = fault-injected sensor  |  clean-sensor health ~= 1.00 (zero corruption)")

    # ── 5. RUL predictions ────────────────────────────────────────────────────
    print(f"\n[5/6]  RUL predictions")

    baseline = mock_lstm_rul(
        clean, clean, true_rul, np.random.default_rng(SEED)
    )

    results: list[dict] = []
    for mode in MODES:
        noisy    = corrupted[mode]
        deg_rul  = mock_lstm_rul(noisy, clean, true_rul, np.random.default_rng(SEED + 1))
        rob_rul, w, hs = robust_predict(clean, noisy, true_rul, np.random.default_rng(SEED + 2))
        results.append({"mode": mode, "deg": deg_rul, "rob": rob_rul, "w": w, "hs": hs})

    # ── 6. Comparison table ───────────────────────────────────────────────────
    print(f"\n[6/6]  Final comparison table")
    sep("=")
    print(f"  Ground-truth RUL : {true_rul} cycles")
    print(f"  Baseline (clean) : {baseline:.1f} cycles   "
          f"(|error| = {abs(baseline - true_rul):.1f} cyc)")
    sep("-")
    print(
        f"  {'Fault Mode':<22}{'Degraded RUL':>14}{'Robust RUL':>12}"
        f"{'|Err| deg':>11}{'|Err| rob':>11}{'Improvement':>13}"
    )
    sep("-")
    for r in results:
        err_d = abs(r["deg"] - true_rul)
        err_r = abs(r["rob"] - true_rul)
        impr  = err_d - err_r
        mode  = r["mode"].replace("_", " ")
        print(
            f"  {mode:<22}{r['deg']:>14.1f}{r['rob']:>12.1f}"
            f"{err_d:>11.1f}{err_r:>11.1f}{impr:>+12.1f} cyc"
        )
    sep("-")

    # Per-mode weights on affected sensors
    print("\n  Channel weights assigned by robust engine (MIN_WEIGHT floor = 0.10):")
    print(f"  {'Sensor':<8}" + "".join(f"{r['mode'][:16]:>17}" for r in results))
    sep("-")
    for i in AFFECTED:
        hs_row = "".join(f"{r['hs'][i]:>17.4f}" for r in results)
        w_row  = "".join(f"{r['w'][i]:>17.4f}" for r in results)
        print(f"  {CMAPSS_SENSOR_NAMES[i]:<8}  health  {hs_row}")
        print(f"  {' ' * 8}  weight  {w_row}")
        print()

    sep("=")
    print("  Demo complete.")
    sep("=")


if __name__ == "__main__":
    main()
