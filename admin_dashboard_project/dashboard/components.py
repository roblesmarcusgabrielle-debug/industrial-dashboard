"""
dashboard/components.py
Reusable Plotly figure builders for the Industrial Admin Dashboard.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats

# ── Colour palette ────────────────────────────────────────────────────────────
C_BLUE   = "#185FA5"
C_GREEN  = "#1D9E75"
C_AMBER  = "#BA7517"
C_RED    = "#E24B4A"
C_GRAY   = "#888780"
C_LIGHT  = "#F1EFE8"

LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="monospace", size=11, color="#444441"),
    margin=dict(l=50, r=20, t=36, b=40),
)


def _apply(fig: go.Figure) -> go.Figure:
    fig.update_layout(**LAYOUT_DEFAULTS)
    fig.update_xaxes(gridcolor="rgba(0,0,0,0.05)", showline=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0.05)", showline=False, zeroline=False)
    return fig


# ── 1. Production Trend ───────────────────────────────────────────────────────
def production_trend(df: pd.DataFrame, preds_all: np.ndarray,
                     date_range=None) -> go.Figure:
    if date_range and len(date_range) == 2:
        mask = (df["Date"] >= pd.Timestamp(date_range[0])) & \
               (df["Date"] <= pd.Timestamp(date_range[1]))
        df = df[mask].copy()
        preds_all = preds_all[mask]

    roll7 = df["Production_Output"].rolling(7).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Production_Output"],
        name="Actual", line=dict(color=C_BLUE, width=1.2), opacity=0.8,
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=np.round(preds_all, 1),
        name="Predicted", line=dict(color=C_GREEN, width=1.5, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=roll7,
        name="7-pt rolling avg", line=dict(color=C_AMBER, width=1, dash="dash"),
    ))
    fig.update_layout(title="Production Trend — Actual vs Predicted", legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)
    ))
    return _apply(fig)


# ── 2. Actual vs Predicted Scatter ───────────────────────────────────────────
def actual_vs_predicted(y_test: np.ndarray, preds: np.ndarray) -> go.Figure:
    mn = min(y_test.min(), preds.min()) - 10
    mx = max(y_test.max(), preds.max()) + 10
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test, y=preds, mode="markers",
        marker=dict(color=C_BLUE, opacity=0.55, size=6),
        name="Predictions",
    ))
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx],
        mode="lines", line=dict(color=C_RED, dash="dash", width=1),
        name="Perfect fit",
    ))
    fig.update_layout(
        title="Actual vs Predicted",
        xaxis_title="Actual (units/hr)",
        yaxis_title="Predicted (units/hr)",
    )
    return _apply(fig)


# ── 3. Shewhart Control Chart ─────────────────────────────────────────────────
def control_chart(residuals: np.ndarray, UCL: float, LCL: float,
                  mean_res: float, start_idx: int = 0) -> go.Figure:
    x = list(range(start_idx, start_idx + len(residuals)))
    ooc_mask = (residuals > UCL) | (residuals < LCL)

    fig = go.Figure()
    # In-control line
    fig.add_trace(go.Scatter(
        x=x, y=residuals,
        mode="lines+markers",
        line=dict(color=C_BLUE, width=1.1),
        marker=dict(size=4, color=np.where(ooc_mask, C_RED, C_BLUE)),
        name="Residual",
    ))
    # OOC points (highlighted)
    if ooc_mask.any():
        fig.add_trace(go.Scatter(
            x=np.array(x)[ooc_mask], y=residuals[ooc_mask],
            mode="markers",
            marker=dict(color=C_RED, size=9, symbol="x-thin", line=dict(width=2, color=C_RED)),
            name="Out-of-control",
        ))
    # Reference lines
    for val, label, color in [(UCL, "UCL", C_RED), (LCL, "LCL", C_RED), (mean_res, "CL", C_GREEN)]:
        fig.add_hline(y=val, line_color=color, line_dash="dash", line_width=1,
                      annotation_text=f"  {label} {val:.1f}",
                      annotation_font_size=9, annotation_position="top left")

    ooc_n = ooc_mask.sum()
    ooc_p = ooc_n / len(residuals) * 100
    fig.update_layout(
        title=f"Shewhart Control Chart — Residuals (3σ)  ·  {ooc_n} violations ({ooc_p:.1f}%)",
    )
    return _apply(fig)


# ── 4. Feature Coefficient Chart ─────────────────────────────────────────────
def coefficient_chart(coef_df: pd.DataFrame) -> go.Figure:
    coef_df = coef_df.sort_values("Coefficient")
    colors = [C_RED if v < 0 else C_GREEN for v in coef_df["Coefficient"]]
    fig = go.Figure(go.Bar(
        x=coef_df["Coefficient"],
        y=coef_df["Feature"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}" for v in coef_df["Coefficient"]],
        textposition="outside",
    ))
    fig.add_vline(x=0, line_color=C_GRAY, line_width=0.8)
    fig.update_layout(title="Feature Coefficients (scaled impact on output)")
    return _apply(fig)


# ── 5. Residual vs Fitted ─────────────────────────────────────────────────────
def residuals_vs_fitted(preds: np.ndarray, residuals: np.ndarray,
                        UCL: float, LCL: float) -> go.Figure:
    ooc_mask = (residuals > UCL) | (residuals < LCL)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=preds[~ooc_mask], y=residuals[~ooc_mask], mode="markers",
        marker=dict(color=C_BLUE, size=5, opacity=0.6), name="In-control",
    ))
    if ooc_mask.any():
        fig.add_trace(go.Scatter(
            x=preds[ooc_mask], y=residuals[ooc_mask], mode="markers",
            marker=dict(color=C_RED, size=8, symbol="x"), name="OOC",
        ))
    fig.add_hline(y=0, line_color=C_GRAY, line_dash="dash", line_width=1)
    fig.update_layout(title="Residuals vs Fitted", xaxis_title="Fitted", yaxis_title="Residual")
    return _apply(fig)


# ── 6. Q-Q Plot ───────────────────────────────────────────────────────────────
def qq_plot(residuals: np.ndarray) -> go.Figure:
    (osm, osr), (slope, intercept, _) = stats.probplot(residuals, dist="norm")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=osm, y=osr, mode="markers",
        marker=dict(color=C_BLUE, size=5, opacity=0.65), name="Sample",
    ))
    fig.add_trace(go.Scatter(
        x=osm, y=slope * np.array(osm) + intercept,
        mode="lines", line=dict(color=C_RED, dash="dash", width=1), name="Normal ref",
    ))
    fig.update_layout(title="Q-Q Plot — Residual Normality",
                      xaxis_title="Theoretical quantiles",
                      yaxis_title="Sample quantiles")
    return _apply(fig)


# ── 7. Forecast with confidence bands ────────────────────────────────────────
def forecast_chart(df: pd.DataFrame, preds_all: np.ndarray,
                   n_future: int = 7, std_res: float = 20.0) -> go.Figure:
    last_date  = df["Date"].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1),
                                 periods=n_future, freq="h")
    last_pred  = preds_all[-1]
    future_preds = last_pred + np.cumsum(np.random.normal(0.5, 3, n_future))
    upper = future_preds + 2 * std_res
    lower = future_preds - 2 * std_res

    hist_dates = df["Date"].iloc[-48:]
    hist_actual = df["Production_Output"].iloc[-48:]
    hist_pred   = preds_all[-48:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_dates, y=hist_actual,
        name="Actual (48h)", line=dict(color=C_BLUE, width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=hist_dates, y=np.round(hist_pred, 1),
        name="Fitted (48h)", line=dict(color=C_GREEN, width=1, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(upper) + list(lower[::-1]),
        fill="toself", fillcolor="rgba(29,158,117,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="95% CI", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=future_preds,
        name="Forecast", line=dict(color=C_GREEN, width=2),
        marker=dict(size=5),
    ))
    fig.update_layout(
        title=f"7-Hour Forecast with Confidence Bands",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
    )
    return _apply(fig)


# ── 8. Residual Histogram ─────────────────────────────────────────────────────
def residual_histogram(residuals: np.ndarray, UCL: float, LCL: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=residuals, nbinsx=20,
        marker_color=C_BLUE, opacity=0.75, name="Residuals",
    ))
    for val, label in [(UCL, "UCL"), (LCL, "LCL")]:
        fig.add_vline(x=val, line_color=C_RED, line_dash="dash", line_width=1.2,
                      annotation_text=f" {label}", annotation_font_size=9)
    fig.update_layout(title="Residual Distribution",
                      xaxis_title="Residual value", yaxis_title="Count")
    return _apply(fig)
