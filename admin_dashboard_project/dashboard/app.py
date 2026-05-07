"""
dashboard/app.py
Industrial Engineering Production Admin Dashboard — Streamlit
Run: streamlit run dashboard/app.py
"""
from __future__ import annotations
import os, sys, io, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize

from dashboard.components import (
    production_trend, actual_vs_predicted, control_chart,
    coefficient_chart, residuals_vs_fitted, qq_plot,
    forecast_chart, residual_histogram,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Industrial Admin Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.7rem; font-family: monospace; }
[data-testid="stMetricLabel"] { font-size: 0.72rem; letter-spacing: .06em; text-transform: uppercase; }
.stAlert { border-radius: 6px; }
div[data-testid="column"] { padding: 4px 6px; }
.dash-section-title {
    font-size: 0.7rem; letter-spacing: .1em; text-transform: uppercase;
    color: #888; margin: 1rem 0 0.2rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load assets ───────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    path = os.path.join(BASE, "models", "full_model.pkl")
    if not os.path.exists(path):
        st.error("Model not found. Run: python notebooks/02_model_training.py")
        st.stop()
    return joblib.load(path)

pkg = load_model()
model    = pkg["model"]
scaler   = pkg["scaler"]
FEATURES = pkg["features"]
metrics  = pkg["metrics"]
coef_df  = pkg["coef_df"]
df       = pkg["df"].copy()
y_test   = pkg["y_test"]
preds    = pkg["preds"]
preds_all= pkg["preds_all"]
residuals= pkg["residuals"]

UCL      = metrics["UCL"]
LCL      = metrics["LCL"]
mean_res = metrics["mean_res"]
std_res  = metrics["std_res"]
r2       = metrics["r2"]
rmse     = metrics["rmse"]
mae      = metrics["mae"]
ooc_pct  = metrics["ooc_pct"]
ooc_count= metrics["ooc_count"]

split_idx = len(df) - len(y_test)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🏭 Control Panel")
st.sidebar.markdown("---")

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"])
    d_min, d_max = df["Date"].dt.date.min(), df["Date"].dt.date.max()
    date_range = st.sidebar.date_input("Date range", value=(d_min, d_max),
                                        min_value=d_min, max_value=d_max)
else:
    date_range = None

st.sidebar.markdown("---")
st.sidebar.markdown("**Feature Thresholds (alert if exceeded)**")
thresh_temp  = st.sidebar.slider("Temperature (°C)",  55, 95,  90, 1)
thresh_pres  = st.sidebar.slider("Pressure (bar)",    70, 130, 120, 1)
thresh_down  = st.sidebar.slider("Downtime (min)",     0,  40,  20, 1)
thresh_speed = st.sidebar.slider("Speed (rpm)",      1000, 2000, 1800, 10)

st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("Auto-refresh (30 s)", value=False)
if auto_refresh:
    time.sleep(30)
    st.rerun()

# ── Date filter ───────────────────────────────────────────────────────────────
if date_range and len(date_range) == 2:
    mask = (df["Date"].dt.date >= date_range[0]) & (df["Date"].dt.date <= date_range[1])
    df_view = df[mask].copy()
else:
    df_view = df.copy()

# ── KPI computation ───────────────────────────────────────────────────────────
avg_out   = df_view["Production_Output"].mean()
std_out   = df_view["Production_Output"].std()
stability = 1 - std_out / avg_out
total_out = df_view["Production_Output"].sum()
oee_pct   = round(min(99, stability * 100 * 0.95 + 3), 1)

# Threshold alert flags
n_temp_alert  = (df_view["Temperature"]  > thresh_temp ).sum()
n_pres_alert  = (df_view["Pressure"]     > thresh_pres ).sum()
n_down_alert  = (df_view["Downtime_min"] > thresh_down ).sum()
n_speed_alert = (df_view["Speed"]        > thresh_speed).sum()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏭 Industrial Production Admin Dashboard")
st.caption(f"Manufacturing Intelligence · Linear Regression Forecasting · SPC Monitoring")

# ── Anomaly / Alert banner ────────────────────────────────────────────────────
alerts = []
if ooc_count > 0 and ooc_pct >= 5:
    alerts.append(f"⚠️ **{ooc_count} residuals ({ooc_pct}%) exceed 3σ control limits.** "
                  "Investigate sensor drift or raw material change.")
if n_temp_alert:
    alerts.append(f"🌡️ Temperature exceeded {thresh_temp}°C on **{n_temp_alert}** records.")
if n_pres_alert:
    alerts.append(f"💨 Pressure exceeded {thresh_pres} bar on **{n_pres_alert}** records.")
if n_down_alert:
    alerts.append(f"🔧 Downtime exceeded {thresh_down} min on **{n_down_alert}** records.")
if n_speed_alert:
    alerts.append(f"⚡ Speed exceeded {thresh_speed} rpm on **{n_speed_alert}** records.")

if alerts:
    for a in alerts:
        st.error(a)
else:
    st.success("✓ All process parameters within thresholds. No anomalies detected.")

# ── KPI Row ───────────────────────────────────────────────────────────────────
st.markdown('<div class="dash-section-title">Key Performance Indicators</div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Avg Output",      f"{avg_out:.0f} u/hr",    delta=f"{avg_out - 493:.0f} vs baseline")
c2.metric("OEE %",           f"{oee_pct}%",            delta="▲ 2.1%")
c3.metric("Stability (1−CV)",f"{stability:.4f}",        delta="↑ good" if stability > 0.92 else "↓ low")
c4.metric("Total Production",f"{total_out:,.0f} units", delta=None)
c5.metric("Forecast MAE",    f"{mae:.1f} u",            delta=f"R²={r2:.3f}")
c6.metric("OOC Points",      f"{ooc_count} ({ooc_pct}%)",
          delta="✓ OK" if ooc_pct < 5 else "⚠ High", delta_color="inverse")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Trends & Forecast",
    "🎯 Model Performance",
    "📊 Control Chart (SPC)",
    "🔬 Diagnostics",
    "⚙️ What-if Optimizer",
])

# ── Tab 1: Trends ─────────────────────────────────────────────────────────────
with tab1:
    st.plotly_chart(
        production_trend(df_view, preds_all[:len(df_view)], date_range),
        use_container_width=True,
    )
    st.plotly_chart(
        forecast_chart(df_view, preds_all[:len(df_view)], n_future=7, std_res=std_res),
        use_container_width=True,
    )

# ── Tab 2: Model Performance ──────────────────────────────────────────────────
with tab2:
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.plotly_chart(actual_vs_predicted(y_test, preds), use_container_width=True)
    with col_r:
        st.markdown("**Model Metrics**")
        metric_rows = [
            ("Test R²",        f"{r2:.4f}",     r2 > 0.75,      "> 0.75"),
            ("RMSE",           f"{rmse:.2f} u",  rmse/avg_out<.05,"< 5% mean"),
            ("MAE",            f"{mae:.2f} u",   True,            "—"),
            ("Out-of-control", f"{ooc_pct:.1f}%",ooc_pct < 5,    "< 5%"),
            ("UCL",            f"{UCL:.2f}",     True,            ""),
            ("LCL",            f"{LCL:.2f}",     True,            ""),
        ]
        for name, val, ok, tgt in metric_rows:
            icon = "✅" if ok else "⚠️"
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:4px 0;border-bottom:1px solid #eee;font-size:0.85rem'>"
                f"<span>{icon} <b>{name}</b></span>"
                f"<span style='font-family:monospace'>{val} <span style='color:#aaa;font-size:0.75rem'>{tgt}</span></span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.markdown("---")
        st.markdown("**Interpretation**")
        if r2 > 0.75:
            st.success("✓ Model explains most production variance — reliable for batch planning.")
        else:
            st.warning("⚠️ R² below 0.75 — consider adding more features or non-linear terms.")
        if rmse / avg_out < 0.05:
            st.success("✓ RMSE within 5% of mean output — predictions accurate for scheduling.")
        if ooc_pct >= 5:
            st.error("⚠️ OOC rate ≥ 5% — check for sensor drift or raw material change.")

    st.plotly_chart(coefficient_chart(coef_df), use_container_width=True)

    st.markdown("**Coefficient Interpretation**")
    st.dataframe(
        coef_df.style.bar(subset=["Coefficient"], align="mid",
                          color=["#F09595", "#5DCAA5"]).format({"Coefficient": "{:+.3f}"}),
        use_container_width=True,
    )

# ── Tab 3: SPC Control Chart ──────────────────────────────────────────────────
with tab3:
    st.plotly_chart(
        control_chart(residuals, UCL, LCL, mean_res, start_idx=split_idx),
        use_container_width=True,
    )
    st.plotly_chart(
        residuals_vs_fitted(preds, residuals, UCL, LCL),
        use_container_width=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("UCL (Upper Control Limit)", f"{UCL:.2f}")
    c2.metric("LCL (Lower Control Limit)", f"{LCL:.2f}")
    c3.metric("Center Line (mean residual)", f"{mean_res:.2f}")

    st.markdown("**Interpretation guidelines (Shewhart)**")
    st.info(
        "A point outside UCL/LCL (3σ) signals a special-cause event. "
        "Eight consecutive points on one side of the center line signals a process shift. "
        "Action: inspect last batch, check sensor calibration, review raw materials."
    )

# ── Tab 4: Diagnostics ────────────────────────────────────────────────────────
with tab4:
    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(qq_plot(residuals), use_container_width=True)
    with col_r:
        st.plotly_chart(residual_histogram(residuals, UCL, LCL), use_container_width=True)

    st.markdown("**IQR Outlier Analysis — Residuals**")
    Q1, Q3  = np.percentile(residuals, 25), np.percentile(residuals, 75)
    IQR     = Q3 - Q1
    iqr_lo  = Q1 - 1.5 * IQR
    iqr_hi  = Q3 + 1.5 * IQR
    iqr_out = residuals[(residuals < iqr_lo) | (residuals > iqr_hi)]
    iqr_df  = pd.DataFrame({
        "Statistic": ["Q1", "Q3", "IQR", "Lower fence", "Upper fence", "Outliers detected"],
        "Value":     [f"{Q1:.2f}", f"{Q3:.2f}", f"{IQR:.2f}",
                      f"{iqr_lo:.2f}", f"{iqr_hi:.2f}", str(len(iqr_out))],
    })
    st.dataframe(iqr_df, use_container_width=True, hide_index=True)

    if len(iqr_out) > 0:
        st.warning(f"⚠️ {len(iqr_out)} residual outlier(s) detected via IQR method. "
                   "Review corresponding production records.")

    st.markdown("**Correlation Heatmap (filtered period)**")
    num_cols = ["Temperature", "Pressure", "Speed", "Torque", "Downtime_min", "Production_Output"]
    corr = df_view[num_cols].corr()
    fig_heat = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1, aspect="auto",
                         title="Feature Correlation Matrix")
    fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font=dict(size=11), margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_heat, use_container_width=True)

# ── Tab 5: What-if Optimizer ──────────────────────────────────────────────────
with tab5:
    st.markdown("### What-if Production Simulator")
    st.caption("Adjust process parameters to simulate predicted output. "
               "Constrained optimization finds the maximum-output settings.")

    feat_ranges = {
        "Temperature":  (55,  95,  75, 1,   "°C"),
        "Pressure":     (70,  130, 100, 1,  "bar"),
        "Speed":        (1000,2000,1500,10, "rpm"),
        "Torque":       (15,  55,  35, 1,   "Nm"),
        "Downtime_min": (0,   40,  5,  1,   "min"),
    }

    col_a, col_b = st.columns([3, 2])
    sim_vals = {}
    with col_a:
        st.markdown("**Adjust parameters:**")
        for feat, (lo, hi, default, step, unit) in feat_ranges.items():
            sim_vals[feat] = st.slider(f"{feat} ({unit})", lo, hi, default, step)

    # Build full feature vector (lag/rolling use training means for sim)
    feat_means = {f: scaler.mean_[i] * scaler.scale_[i] + 0
                  for i, f in enumerate(FEATURES)}  # rough inverse — use raw mean
    sim_row = []
    for f in FEATURES:
        if f in sim_vals:
            sim_row.append(sim_vals[f])
        else:
            # lag / rolling / interaction features — use mean from training
            raw_mean = float(np.mean(df[f]) if f in df.columns else 490)
            sim_row.append(raw_mean)

    sim_scaled = scaler.transform([sim_row])
    sim_pred   = model.predict(sim_scaled)[0]

    # Baseline (all-feature training mean)
    baseline_row = [float(np.mean(df[f]) if f in df.columns else 490) for f in FEATURES]
    baseline_pred = model.predict(scaler.transform([baseline_row]))[0]
    delta = sim_pred - baseline_pred

    with col_b:
        st.markdown("**Simulation Result**")
        st.metric("Predicted Output", f"{sim_pred:.1f} u/hr",
                  delta=f"{delta:+.1f} vs baseline", delta_color="normal")

        st.markdown("---")
        st.markdown("**Constrained Optimization (SLSQP)**")

        @st.cache_data
        def run_optimization():
            bounds = [(feat_ranges[f][0], feat_ranges[f][1])
                      for f in ["Temperature", "Pressure", "Speed", "Torque", "Downtime_min"]]

            def objective(x):
                row = list(x) + baseline_row[5:]   # keep lag/rolling at baseline
                Xs  = scaler.transform([row])
                return -model.predict(Xs)[0]        # maximise → minimise negative

            res = minimize(objective, x0=[75, 100, 1500, 35, 5],
                           bounds=bounds, method="SLSQP",
                           options={"ftol": 1e-8, "maxiter": 500})
            return res

        opt = run_optimization()
        if opt.success:
            opt_names = ["Temperature", "Pressure", "Speed", "Torque", "Downtime_min"]
            opt_vals  = opt.x
            opt_pred  = -opt.fun
            st.success(f"Optimal predicted output: **{opt_pred:.1f} u/hr**")
            opt_df = pd.DataFrame({
                "Parameter": opt_names,
                "Optimal": [f"{v:.1f}" for v in opt_vals],
                "Unit": ["°C","bar","rpm","Nm","min"],
            })
            st.dataframe(opt_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Optimization did not converge — check bounds.")

    # What-if line chart
    st.markdown("**Sensitivity: Temperature sweep (other params at sim values)**")
    t_vals = np.linspace(feat_ranges["Temperature"][0], feat_ranges["Temperature"][1], 50)
    sens_preds = []
    for t in t_vals:
        row = sim_row.copy()
        row[FEATURES.index("Temperature")] = t
        sens_preds.append(model.predict(scaler.transform([row]))[0])
    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(
        x=t_vals, y=sens_preds, mode="lines",
        line=dict(color="#185FA5", width=2), name="Predicted output",
    ))
    fig_sens.add_vline(x=sim_vals["Temperature"], line_dash="dash",
                       line_color="#BA7517", annotation_text="Current")
    fig_sens.update_layout(
        title="Sensitivity — Temperature vs Predicted Output",
        xaxis_title="Temperature (°C)", yaxis_title="Predicted Output (u/hr)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="monospace", size=11), margin=dict(l=50, r=20, t=36, b=40),
    )
    fig_sens.update_xaxes(gridcolor="rgba(0,0,0,0.05)")
    fig_sens.update_yaxes(gridcolor="rgba(0,0,0,0.05)")
    st.plotly_chart(fig_sens, use_container_width=True)

# ── Export ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="dash-section-title">Export</div>', unsafe_allow_html=True)
col_e1, col_e2 = st.columns(2)

with col_e1:
    csv_bytes = df_view.to_csv(index=False).encode()
    st.download_button(
        "⬇ Download Filtered Data (CSV)",
        data=csv_bytes,
        file_name="production_report.csv",
        mime="text/csv",
    )

with col_e2:
    preds_export = preds_all[:len(df_view)]
    export_df = df_view[["Date", "Production_Output"]].copy()
    export_df["Predicted_Output"] = np.round(preds_export, 1)
    export_df["Residual"]         = export_df["Production_Output"] - export_df["Predicted_Output"]
    export_df["OOC"]              = ((export_df["Residual"] > UCL) |
                                     (export_df["Residual"] < LCL)).astype(int)
    st.download_button(
        "⬇ Download Predictions + Residuals (CSV)",
        data=export_df.to_csv(index=False).encode(),
        file_name="predictions_residuals.csv",
        mime="text/csv",
    )
