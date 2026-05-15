#!/usr/bin/env python3
"""
demo/app.py — Streamlit interactive demo for the RobustPM robustness layer.

Lets you pick any of the 100 FD001 test engines, choose a fault mode and
severity, and see in real time how sensor health scoring and channel
weighting recover RUL prediction accuracy.

Run from the project root:
    streamlit run demo/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── path setup ────────────────────────────────────────────────────────────────
_DEMO_DIR = Path(__file__).resolve().parent          # demo/
_ROOT     = _DEMO_DIR.parent                         # project root
_PM_DIR   = _ROOT / "predictive-maintenance"

sys.path.insert(0, str(_PM_DIR))

from robustness.sensor_degradation import CMAPSS_SENSOR_NAMES, SensorDegradationInjector

# ── constants ─────────────────────────────────────────────────────────────────
MODES = ["gaussian_noise", "stuck_at_value", "partial_dropout", "linear_drift"]
MODE_LABELS = {
    "gaussian_noise":  "Gaussian Noise",
    "stuck_at_value":  "Stuck at Value",
    "partial_dropout": "Partial Dropout",
    "linear_drift":    "Linear Drift",
}
N_SENSORS  = 21
MIN_WEIGHT = 0.1
BIAS_SCALE = 80.0
N_NOISE    = 4.0
SEED_BASE  = 42
WINDOW     = 30   # cycles per engine window

TEST_FILE = _ROOT / "archive" / "CMaps" / "test_FD001.txt"
RUL_FILE  = _ROOT / "archive" / "CMaps" / "RUL_FD001.txt"


# ── cached data loaders ───────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_engines() -> dict[int, np.ndarray]:
    raw   = np.loadtxt(str(TEST_FILE), dtype=np.float64)
    units = raw[:, 0].astype(int)
    out: dict[int, np.ndarray] = {}
    for uid in np.unique(units):
        out[int(uid)] = raw[units == uid, 5:26].astype(np.float64)
    return out


@st.cache_data(show_spinner=False)
def load_true_ruls() -> dict[int, float]:
    vals = np.loadtxt(str(RUL_FILE), dtype=np.float64)
    return {int(i + 1): float(v) for i, v in enumerate(vals)}


# ── numpy-only inference (no TensorFlow required for demo) ────────────────────

def compute_health_scores(clean: np.ndarray, noisy: np.ndarray) -> np.ndarray:
    err = ((noisy - clean) ** 2).mean(axis=0)
    thr = np.maximum(np.var(clean, axis=0), 1e-10)
    return 1.0 - np.clip(err / thr, 0.0, 1.0)


def compute_weights(health: np.ndarray) -> np.ndarray:
    raw = np.maximum(health, MIN_WEIGHT)
    return raw * (N_SENSORS / raw.sum())


def mock_lstm(
    data: np.ndarray, ref: np.ndarray, true_rul: float, rng: np.random.Generator
) -> float:
    dev = float(np.sqrt(np.mean((data - ref) ** 2)))
    return float(np.clip(true_rul - dev * BIAS_SCALE + rng.normal(0.0, N_NOISE), 1.0, 350.0))


# ── page configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RobustPM — Robustness Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("RobustPM Controls")
    st.caption("NASA C-MAPSS FD001 · 100 test engines · 21 sensors")
    st.divider()

    engine_id = st.slider("Engine ID", min_value=1, max_value=100, value=1)

    fault_mode = st.selectbox(
        "Fault Mode",
        options=MODES,
        format_func=lambda m: MODE_LABELS[m],
    )

    severity = st.slider(
        "Fault Severity",
        min_value=0.0, max_value=1.0, value=0.5, step=0.1,
        help="0 = no corruption · 1 = maximum corruption",
    )

    affected_names = st.multiselect(
        "Affected Sensors",
        options=CMAPSS_SENSOR_NAMES,
        default=["T30", "P15", "Ps30"],
        help="Sensors that will receive the fault injection",
    )

    st.divider()
    run = st.button("Run Analysis", type="primary", use_container_width=True)

# ── page header ───────────────────────────────────────────────────────────────

st.title("RobustPM — Sensor Robustness Layer Demo")
st.caption(
    "Inject a synthetic sensor fault, observe health scores degrade, "
    "and see how channel-weighted inference recovers RUL accuracy."
)

# ── guard: need at least one sensor selected ───────────────────────────────────

if not affected_names:
    st.warning("Select at least one affected sensor in the sidebar, then click Run Analysis.")
    st.stop()

# ── run computation ───────────────────────────────────────────────────────────

if run:
    with st.spinner("Running analysis..."):
        engines   = load_engines()
        true_ruls = load_true_ruls()

        sensors_full = engines[engine_id]
        true_rul     = true_ruls[engine_id]
        n_cycles     = sensors_full.shape[0]

        # Last WINDOW cycles, z-score normalised
        win     = sensors_full[max(0, n_cycles - WINDOW):]
        ch_mean = win.mean(axis=0)
        ch_std  = np.where(win.std(axis=0) < 1e-8, 1.0, win.std(axis=0))
        clean   = (win - ch_mean) / ch_std

        affected_indices = [CMAPSS_SENSOR_NAMES.index(n) for n in affected_names]

        # Fault injection
        injector = SensorDegradationInjector(
            fault_severity=severity,
            affected_sensor_indices=affected_indices,
            random_seed=SEED_BASE,
        )
        noisy = injector.inject(clean, fault_mode)

        # Health scoring and channel weighting
        health  = compute_health_scores(clean, noisy)
        weights = compute_weights(health)

        w_noisy = noisy * weights[np.newaxis, :]
        w_clean = clean * weights[np.newaxis, :]

        # RUL predictions (fixed seeds for reproducibility)
        clean_rul = mock_lstm(clean,   clean,   true_rul, np.random.default_rng(SEED_BASE))
        deg_rul   = mock_lstm(noisy,   clean,   true_rul, np.random.default_rng(SEED_BASE + 1))
        rob_rul   = mock_lstm(w_noisy, w_clean, true_rul, np.random.default_rng(SEED_BASE + 2))

        deg_abs = abs(deg_rul - true_rul)
        rob_abs = abs(rob_rul - true_rul)
        recovery_pct = (
            (deg_abs - rob_abs) / deg_abs * 100.0 if deg_abs > 1e-6 else 100.0
        )

        st.session_state["results"] = dict(
            engine_id=engine_id,
            true_rul=true_rul,
            clean_rul=clean_rul,
            deg_rul=deg_rul,
            rob_rul=rob_rul,
            health=health,
            weights=weights,
            clean=clean,
            noisy=noisy,
            affected_indices=affected_indices,
            affected_names=affected_names,
            fault_mode=fault_mode,
            severity=severity,
            recovery_pct=recovery_pct,
            n_cycles=n_cycles,
        )

if "results" not in st.session_state:
    st.info("Configure the controls in the sidebar and click **Run Analysis** to begin.")
    st.stop()

r = st.session_state["results"]

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — Sensor Health Scores
# ═══════════════════════════════════════════════════════════════════════════════

st.subheader("1. Sensor Health Scores")

health          = r["health"]
affected_idx    = r["affected_indices"]
affected_set    = set(affected_idx)

# Bar colours: green / amber / red
bar_colors = [
    "#27ae60" if s > 0.8 else ("#f39c12" if s >= 0.5 else "#e74c3c")
    for s in health
]

# Mark affected sensors with a bullet in the label
y_labels = [
    f"• {CMAPSS_SENSOR_NAMES[i]}" if i in affected_set else CMAPSS_SENSOR_NAMES[i]
    for i in range(N_SENSORS)
]

fig_health = go.Figure(
    go.Bar(
        x=health,
        y=y_labels,
        orientation="h",
        marker_color=bar_colors,
        text=[f"{s:.2f}" for s in health],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    )
)
fig_health.add_vline(
    x=0.8, line_dash="dash", line_color="#27ae60", line_width=1.5,
    annotation_text="Healthy threshold (0.8)",
    annotation_position="top right",
    annotation_font_size=11,
)
fig_health.add_vline(
    x=0.5, line_dash="dash", line_color="#e74c3c", line_width=1.5,
    annotation_text="Degraded threshold (0.5)",
    annotation_position="bottom right",
    annotation_font_size=11,
)
fig_health.update_layout(
    height=480,
    xaxis=dict(range=[0, 1.22], title="Health Score"),
    yaxis=dict(title="", autorange="reversed"),
    plot_bgcolor="white",
    margin=dict(l=10, r=60, t=30, b=40),
    annotations=[
        dict(
            text="• = fault-affected sensor",
            xref="paper", yref="paper",
            x=0.0, y=-0.07,
            showarrow=False,
            font=dict(size=11, color="#555"),
        )
    ],
)
fig_health.update_xaxes(showgrid=True, gridcolor="#eee")

st.plotly_chart(fig_health, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — RUL Comparison
# ═══════════════════════════════════════════════════════════════════════════════

st.subheader("2. RUL Comparison")

true_rul = r["true_rul"]
deg_rul  = r["deg_rul"]
rob_rul  = r["rob_rul"]
rec_pct  = r["recovery_pct"]

deg_err_pct = abs(deg_rul - true_rul) / max(true_rul, 1.0) * 100.0
rob_err_pct = abs(rob_rul - true_rul) / max(true_rul, 1.0) * 100.0

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="True RUL",
        value=f"{true_rul:.0f} cycles",
        help="Ground truth remaining useful life from RUL_FD001.txt",
    )
    st.caption(f"≈ {true_rul * 0.5:.0f} hours")

with col2:
    st.metric(
        label="Degraded RUL",
        value=f"{deg_rul:.0f} cycles",
        delta=f"{deg_rul - true_rul:+.1f} vs true",
        delta_color="inverse",
    )
    if deg_err_pct > 10:
        st.error(f"Error: **{deg_err_pct:.1f}%** — exceeds 10% threshold")
    else:
        st.success(f"Error: **{deg_err_pct:.1f}%** — within threshold")

with col3:
    st.metric(
        label="Robust RUL",
        value=f"{rob_rul:.0f} cycles",
        delta=f"{rob_rul - true_rul:+.1f} vs true",
        delta_color="inverse",
    )
    if rob_err_pct <= 5:
        st.success(f"Error: **{rob_err_pct:.1f}%** — within 5%")
    elif rob_err_pct <= 15:
        st.warning(f"Error: **{rob_err_pct:.1f}%**")
    else:
        st.error(f"Error: **{rob_err_pct:.1f}%**")

# Gauge — error recovery percentage
gauge_bar_color = (
    "#27ae60" if rec_pct >= 70 else ("#f39c12" if rec_pct >= 30 else "#e74c3c")
)

fig_gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=rec_pct,
        number={"suffix": "%", "font": {"size": 40, "color": gauge_bar_color}},
        title={"text": "Error Recovery", "font": {"size": 15}},
        gauge={
            "axis": {"range": [-50, 100], "ticksuffix": "%", "tickfont": {"size": 11}},
            "bar": {"color": gauge_bar_color, "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 1,
            "bordercolor": "#ccc",
            "steps": [
                {"range": [-50, 30],  "color": "#fdecea"},
                {"range": [30,  70],  "color": "#fff8e1"},
                {"range": [70, 100],  "color": "#e8f5e9"},
            ],
        },
    )
)
fig_gauge.update_layout(
    height=260,
    margin=dict(l=40, r=40, t=60, b=10),
)
st.plotly_chart(fig_gauge, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — Fault Injection Detail
# ═══════════════════════════════════════════════════════════════════════════════

st.subheader("3. Fault Injection Detail")

clean       = r["clean"]
noisy       = r["noisy"]
first_idx   = r["affected_indices"][0]
sensor_name = CMAPSS_SENSOR_NAMES[first_idx]
n_win       = clean.shape[0]
cycles      = list(range(1, n_win + 1))

fig_inj = go.Figure()
fig_inj.add_trace(
    go.Scatter(
        x=cycles,
        y=clean[:, first_idx].tolist(),
        name="Clean signal",
        line=dict(color="royalblue", width=2),
        mode="lines+markers",
        marker=dict(size=4),
    )
)
fig_inj.add_trace(
    go.Scatter(
        x=cycles,
        y=noisy[:, first_idx].tolist(),
        name=f"Degraded — {MODE_LABELS[r['fault_mode']]} (sev={r['severity']:.1f})",
        line=dict(color="tomato", width=2, dash="dot"),
        mode="lines+markers",
        marker=dict(size=4),
    )
)
fig_inj.update_layout(
    title=dict(
        text=f"Sensor {sensor_name}: clean vs degraded  "
             f"(Engine {r['engine_id']}, last {n_win} cycles)",
        font=dict(size=14),
    ),
    xaxis_title="Cycle (within window)",
    yaxis_title="Normalised value (z-score)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    plot_bgcolor="white",
    height=350,
    margin=dict(l=10, r=10, t=60, b=40),
)
fig_inj.update_xaxes(showgrid=True, gridcolor="#eee")
fig_inj.update_yaxes(showgrid=True, gridcolor="#eee")

st.plotly_chart(fig_inj, use_container_width=True)

# ── footer metadata ───────────────────────────────────────────────────────────

st.divider()
st.caption(
    f"Engine **{r['engine_id']}** | True RUL: **{r['true_rul']:.0f} cycles** | "
    f"Mode: **{MODE_LABELS[r['fault_mode']]}** @ severity **{r['severity']:.1f}** | "
    f"Affected: **{r['affected_names']}**"
)
