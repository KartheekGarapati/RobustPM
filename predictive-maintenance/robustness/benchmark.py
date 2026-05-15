#!/usr/bin/env python3
"""
benchmark.py — Robustness layer evaluation across all 100 FD001 test engines.

For each engine in test_FD001.txt:
  1. Loads the last 30 cycles of its sensor sequence.
  2. Reads true RUL from RUL_FD001.txt.
  3. Injects all 4 fault modes on sensors T30/P15/Ps30 at severity 0.50.
  4. Computes clean, degraded, and robust RUL predictions.
  5. Appends one row per (engine, fault_mode) to benchmark_results.csv.

Generates benchmark_results.png: 4 paired box-plot subplots (one per fault
mode) showing degraded error vs robust error distributions across 100 engines.

Usage
-----
From the project root:
    python predictive-maintenance/robustness/benchmark.py

From predictive-maintenance/:
    python robustness/benchmark.py

All outputs land in the same directory as this script (robustness/).
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_BENCH_DIR = Path(__file__).resolve().parent        # .../robustness/
_PM_DIR    = _BENCH_DIR.parent                      # .../predictive-maintenance/
_ROOT      = _PM_DIR.parent                         # project root

sys.path.insert(0, str(_PM_DIR))
from robustness.sensor_degradation import SensorDegradationInjector, CMAPSS_SENSOR_NAMES

# ── configuration ─────────────────────────────────────────────────────────────
TEST_FILE  = _ROOT / "archive" / "CMaps" / "test_FD001.txt"
RUL_FILE   = _ROOT / "archive" / "CMaps" / "RUL_FD001.txt"
CSV_OUT    = _BENCH_DIR / "benchmark_results.csv"
PNG_OUT    = _BENCH_DIR / "benchmark_results.png"

AFFECTED   = [2, 5, 10]   # T30, P15, Ps30
SEVERITY   = 0.5
SEED_BASE  = 42
WINDOW     = 30            # last N cycles used per engine
MODES      = [
    "gaussian_noise",
    "stuck_at_value",
    "partial_dropout",
    "linear_drift",
]
N_SENSORS  = 21
MIN_WEIGHT = 0.1
BIAS_SCALE = 80.0          # cycles per normalised L2 deviation unit
N_NOISE    = 4.0           # Gaussian noise on RUL mock (cycles)
W          = 70            # print width


# ── print helpers ─────────────────────────────────────────────────────────────

def sep(char: str = "-", width: int = W) -> None:
    print(char * width)


# ── data loading ──────────────────────────────────────────────────────────────

def load_test_engines(path: Path) -> dict[int, np.ndarray]:
    """
    Parse test_FD001.txt into a dict {engine_id: sensor_array}.

    For each engine returns shape (n_cycles, 21) float64 containing only the
    21 sensor columns (columns 5-25).  Engines are keyed by their 1-based ID.
    """
    raw   = np.loadtxt(str(path), dtype=np.float64)
    units = raw[:, 0].astype(int)
    engines: dict[int, np.ndarray] = {}
    for uid in np.unique(units):
        mask = units == uid
        engines[int(uid)] = raw[mask, 5:26].astype(np.float64)
    return engines


def load_true_rul(path: Path) -> dict[int, float]:
    """
    Parse RUL_FD001.txt into {engine_id: true_rul}.

    File has one float per line; engine IDs are 1-based in order.
    """
    rul_vals = np.loadtxt(str(path), dtype=np.float64)
    return {int(i + 1): float(v) for i, v in enumerate(rul_vals)}


# ── numpy-only health scoring (mirrors production SensorHealthMonitor) ────────

def compute_health_scores(clean: np.ndarray, noisy: np.ndarray) -> np.ndarray:
    """
    Per-sensor health score in [0, 1] via normalised reconstruction error.

    Approximates the autoencoder by measuring deviation from clean baseline:
        error[i]     = MSE(noisy[:, i] - clean[:, i])
        threshold[i] = Var(clean[:, i])
        health[i]    = 1 - clip(error / threshold, 0, 1)

    Handles zero-variance channels (constant sensors) via a 1e-10 floor.
    """
    err = ((noisy - clean) ** 2).mean(axis=0)    # (21,)
    thr = np.var(clean, axis=0)
    thr = np.maximum(thr, 1e-10)
    return 1.0 - np.clip(err / thr, 0.0, 1.0)


# ── numpy-only channel weighting (mirrors RobustInferenceEngine) ──────────────

def compute_weights(health: np.ndarray) -> np.ndarray:
    """
    Convert health scores to mean-normalised channel weights.

        raw[i]    = max(health[i], MIN_WEIGHT)
        w_norm[i] = raw[i] * N / sum(raw)
    """
    raw = np.maximum(health, MIN_WEIGHT)
    return raw * (N_SENSORS / raw.sum())


# ── mock LSTM (same physics-motivated estimator as demo_robustness.py) ────────

def mock_lstm(
    data: np.ndarray,
    reference: np.ndarray,
    true_rul: float,
    rng: np.random.Generator,
) -> float:
    """
    Simulate LSTM RUL prediction.

    Corruption moves data away from reference; the resulting L2 distance is
    scaled to a RUL bias.  A small Gaussian perturbation models model variance.
    """
    dev   = float(np.sqrt(np.mean((data - reference) ** 2)))
    bias  = dev * BIAS_SCALE
    noise = float(rng.normal(0.0, N_NOISE))
    return float(np.clip(true_rul - bias + noise, 1.0, 350.0))


# ── per-engine evaluation ─────────────────────────────────────────────────────

def evaluate_engine(
    engine_id: int,
    sensors_full: np.ndarray,
    true_rul: float,
) -> list[dict]:
    """
    Run all 4 fault modes on one engine and return a list of result dicts.

    Uses the last WINDOW cycles of the engine's sensor sequence (or all
    cycles if fewer than WINDOW are available).
    """
    # Take the last WINDOW cycles
    n = sensors_full.shape[0]
    window = sensors_full[max(0, n - WINDOW) :]   # (≤30, 21)

    # Z-score normalise using window statistics
    ch_mean = window.mean(axis=0)
    ch_std  = window.std(axis=0)
    ch_std  = np.where(ch_std < 1e-8, 1.0, ch_std)
    clean   = (window - ch_mean) / ch_std          # (≤30, 21)

    # Baseline clean RUL (same seed per engine for reproducibility)
    clean_rul = mock_lstm(
        clean, clean, true_rul,
        np.random.default_rng(SEED_BASE + engine_id)
    )

    results = []
    for m_idx, mode in enumerate(MODES):
        # Deterministic seeds per (engine, mode) combination
        inj_seed  = SEED_BASE + 1000 * (m_idx + 1) + engine_id
        deg_seed  = SEED_BASE + 2000 * (m_idx + 1) + engine_id
        rob_seed  = SEED_BASE + 3000 * (m_idx + 1) + engine_id

        # Inject fault
        injector = SensorDegradationInjector(
            fault_severity=SEVERITY,
            affected_sensor_indices=AFFECTED,
            random_seed=inj_seed,
        )
        noisy = injector.inject(clean, mode)       # (≤30, 21)

        # Degraded prediction
        deg_rul = mock_lstm(noisy, clean, true_rul, np.random.default_rng(deg_seed))

        # Robust prediction: down-weight degraded sensors
        health   = compute_health_scores(clean, noisy)
        weights  = compute_weights(health)
        w_noisy  = noisy * weights[np.newaxis, :]
        w_clean  = clean * weights[np.newaxis, :]
        rob_rul  = mock_lstm(w_noisy, w_clean, true_rul, np.random.default_rng(rob_seed))

        deg_err = deg_rul - true_rul               # signed error
        rob_err = rob_rul - true_rul
        abs_deg = abs(deg_err)
        abs_rob = abs(rob_err)
        recovery = (
            round((abs_deg - abs_rob) / abs_deg * 100, 2)
            if abs_deg > 1e-6 else 100.0
        )

        results.append({
            "engine_id":     engine_id,
            "fault_mode":    mode,
            "true_rul":      round(true_rul, 2),
            "clean_rul":     round(clean_rul, 2),
            "degraded_rul":  round(deg_rul, 2),
            "robust_rul":    round(rob_rul, 2),
            "degraded_error": round(deg_err, 2),
            "robust_error":   round(rob_err, 2),
            "recovery_pct":  recovery,
        })
    return results


# ── CSV writer ────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "engine_id", "fault_mode", "true_rul", "clean_rul",
    "degraded_rul", "robust_rul", "degraded_error", "robust_error",
    "recovery_pct",
]


def write_csv(records: list[dict], path: Path) -> None:
    with open(str(path), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)


# ── plotting ──────────────────────────────────────────────────────────────────

def generate_plot(records: list[dict], path: Path) -> None:
    """
    4-panel figure: one paired box plot per fault mode.

    Each panel shows |degraded error| vs |robust error| across 100 engines
    so the distribution shift is immediately visible.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend, safe on all platforms
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  [plot] matplotlib not available — skipping PNG generation")
        return

    MODE_LABELS = {
        "gaussian_noise":  "Gaussian Noise",
        "stuck_at_value":  "Stuck at Value",
        "partial_dropout": "Partial Dropout",
        "linear_drift":    "Linear Drift",
    }
    COLORS = {"Degraded": "#e07b54", "Robust": "#5b9bd5"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "RobustPM — FD001 Benchmark: |RUL Error| across 100 Test Engines\n"
        f"Fault severity={SEVERITY}  |  Affected sensors: "
        f"{[CMAPSS_SENSOR_NAMES[i] for i in AFFECTED]}",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    for ax, mode in zip(axes.flat, MODES):
        rows     = [r for r in records if r["fault_mode"] == mode]
        deg_abs  = [abs(r["degraded_error"]) for r in rows]
        rob_abs  = [abs(r["robust_error"]) for r in rows]
        mean_rec = float(np.mean([r["recovery_pct"] for r in rows]))

        bp = ax.boxplot(
            [deg_abs, rob_abs],
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker="o", markersize=3, alpha=0.5),
        )
        bp["boxes"][0].set_facecolor(COLORS["Degraded"])
        bp["boxes"][1].set_facecolor(COLORS["Robust"])

        ax.set_title(MODE_LABELS[mode], fontsize=12, fontweight="bold")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Degraded", "Robust"], fontsize=11)
        ax.set_ylabel("|RUL Error| (cycles)", fontsize=10)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.text(
            0.97, 0.97,
            f"Mean recovery: {mean_rec:.1f}%",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.8),
        )

        # Annotate median values
        for pos, data, label in [(1, deg_abs, "D"), (2, rob_abs, "R")]:
            med = float(np.median(data))
            ax.text(pos, med + 0.5, f"{med:.1f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="black")

    legend_patches = [
        mpatches.Patch(color=COLORS["Degraded"], label="Degraded (no correction)"),
        mpatches.Patch(color=COLORS["Robust"],   label="Robust (health-weighted)"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=2,
        fontsize=11,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.03),
    )

    plt.tight_layout()
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved -> {path}")


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary(records: list[dict]) -> None:
    sep("=")
    print("  FD001 Benchmark Summary  —  100 engines x 4 fault modes = 400 evaluations")
    sep("=")
    print(f"  Fault severity : {SEVERITY}")
    print(f"  Affected sensors: {[CMAPSS_SENSOR_NAMES[i] for i in AFFECTED]}  (indices {AFFECTED})")
    print(f"  Window length  : last {WINDOW} cycles per engine")
    sep("-")

    header = (
        f"  {'Fault Mode':<22}"
        f"{'Mean |Err| Deg':>16}"
        f"{'Mean |Err| Rob':>16}"
        f"{'Median Recovery':>17}"
        f"{'Mean Recovery':>15}"
        f"{'Engines Improved':>18}"
    )
    print(header)
    sep("-")

    overall_deg, overall_rob, overall_rec = [], [], []
    for mode in MODES:
        rows     = [r for r in records if r["fault_mode"] == mode]
        deg_abs  = [abs(r["degraded_error"]) for r in rows]
        rob_abs  = [abs(r["robust_error"])   for r in rows]
        rec      = [r["recovery_pct"]        for r in rows]
        improved = sum(1 for d, rb in zip(deg_abs, rob_abs) if rb < d)

        overall_deg.extend(deg_abs)
        overall_rob.extend(rob_abs)
        overall_rec.extend(rec)

        print(
            f"  {mode.replace('_', ' '):<22}"
            f"{np.mean(deg_abs):>16.2f}"
            f"{np.mean(rob_abs):>16.2f}"
            f"{np.median(rec):>17.1f}%"
            f"{np.mean(rec):>14.1f}%"
            f"{improved:>15}/100"
        )

    sep("-")
    print(
        f"  {'OVERALL':<22}"
        f"{np.mean(overall_deg):>16.2f}"
        f"{np.mean(overall_rob):>16.2f}"
        f"{np.median(overall_rec):>17.1f}%"
        f"{np.mean(overall_rec):>14.1f}%"
        f"{sum(1 for d,r in zip(overall_deg, overall_rob) if r<d):>15}/400"
    )
    sep("=")

    # Distribution statistics
    print("\n  Absolute error distribution across all engines and modes:")
    sep("-")
    print(f"  {'Metric':<28} {'Degraded':>12} {'Robust':>12}")
    sep("-")
    for label, fn in [
        ("Min |error| (cycles)",    np.min),
        ("25th pct |error|",        lambda x: np.percentile(x, 25)),
        ("Median |error| (cycles)", np.median),
        ("75th pct |error|",        lambda x: np.percentile(x, 75)),
        ("Mean |error| (cycles)",   np.mean),
        ("Max |error| (cycles)",    np.max),
        ("Std |error| (cycles)",    np.std),
    ]:
        print(
            f"  {label:<28}"
            f"{fn(overall_deg):>12.2f}"
            f"{fn(overall_rob):>12.2f}"
        )
    sep("=")

    # Best and worst engines per mode
    print("\n  Top-5 most-improved engines (per mode, by absolute error reduction):")
    sep("-")
    for mode in MODES:
        rows = sorted(
            [r for r in records if r["fault_mode"] == mode],
            key=lambda r: abs(r["degraded_error"]) - abs(r["robust_error"]),
            reverse=True,
        )[:5]
        ids   = [str(r["engine_id"]) for r in rows]
        gains = [f"{abs(r['degraded_error']) - abs(r['robust_error']):.1f}" for r in rows]
        pairs = "  ".join(f"E{i}({g}cyc)" for i, g in zip(ids, gains))
        print(f"  {mode.replace('_', ' '):<22}  {pairs}")
    sep("-")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.perf_counter()

    sep("=")
    print("  RobustPM — FD001 Robustness Benchmark")
    sep("=")
    print(f"  Test data  : {TEST_FILE}")
    print(f"  RUL data   : {RUL_FILE}")
    print(f"  CSV output : {CSV_OUT}")
    print(f"  PNG output : {PNG_OUT}")
    sep("-")

    # Load data
    print("\n  Loading test data...")
    engines  = load_test_engines(TEST_FILE)
    true_ruls = load_true_rul(RUL_FILE)
    n_engines = len(engines)
    print(f"  {n_engines} engines loaded | "
          f"RUL range: {min(true_ruls.values()):.0f}–{max(true_ruls.values()):.0f} cycles "
          f"(mean {np.mean(list(true_ruls.values())):.1f})")

    # Evaluate all engines
    print(f"\n  Evaluating {n_engines} engines x {len(MODES)} fault modes "
          f"= {n_engines * len(MODES)} predictions...")
    sep("-")

    all_records: list[dict] = []
    for idx, engine_id in enumerate(sorted(engines.keys()), start=1):
        records = evaluate_engine(
            engine_id=engine_id,
            sensors_full=engines[engine_id],
            true_rul=true_ruls[engine_id],
        )
        all_records.extend(records)

        # Progress bar: print every 10 engines
        if idx % 10 == 0 or idx == n_engines:
            done  = idx * len(MODES)
            total = n_engines * len(MODES)
            bar_w = 30
            filled = int(bar_w * idx / n_engines)
            bar   = "#" * filled + "." * (bar_w - filled)
            elapsed = time.perf_counter() - t0
            print(f"  [{bar}] {idx:>3}/{n_engines}  "
                  f"({done}/{total} preds)  {elapsed:.1f}s")

    sep("-")
    print(f"  Benchmark complete in {time.perf_counter() - t0:.2f}s\n")

    # Save CSV
    write_csv(all_records, CSV_OUT)
    print(f"  Results saved -> {CSV_OUT}  ({len(all_records)} rows)")

    # Generate plot
    print("  Generating plot...")
    generate_plot(all_records, PNG_OUT)

    # Print summary
    print()
    print_summary(all_records)


if __name__ == "__main__":
    main()
