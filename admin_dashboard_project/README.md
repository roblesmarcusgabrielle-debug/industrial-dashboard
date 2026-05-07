# Industrial Engineering Production Admin Dashboard

> A decision-support tool for plant managers and process engineers.  
> Linear regression forecasting · Statistical Process Control (SPC) · Real-time KPI monitoring

---

## Executive Summary

This dashboard transforms raw manufacturing sensor data into actionable production intelligence.
It combines regression-based output forecasting, Shewhart control charts for anomaly detection,
and a constrained what-if optimizer — giving plant managers a single view to:

- **Reduce downtime** by surfacing out-of-control signals before they cascade into failures.
- **Maximize yield** via the optimizer that finds optimal Temperature / Pressure / Speed / Torque settings.
- **Track performance** against OEE, stability, and forecast accuracy KPIs per shift.

---

## Project Structure

```
admin_dashboard_project/
│
├── data/
│   ├── generate_data.py       # synthetic dataset generator (replace with Kaggle CSV)
│   └── industrial_data.csv    # generated or downloaded dataset
│
├── notebooks/
│   ├── 01_eda_analysis.py     # EDA, correlation, outlier report, plots → models/eda_plots.png
│   └── 02_model_training.py   # Feature engineering, LR training, diagnostics → models/full_model.pkl
│
├── dashboard/
│   ├── app.py                 # main Streamlit dashboard
│   ├── components.py          # reusable Plotly figure builders
│   └── assets/style.css       # (optional) extra CSS overrides
│
├── models/
│   ├── full_model.pkl         # trained LinearRegression + scaler + metadata
│   ├── eda_plots.png          # EDA visualisation grid
│   └── diagnostic_plots.png   # residual diagnostics grid
│
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Environment

```bash
python -m venv venv
source venv/bin/activate          # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Data

**Option A — use the included synthetic dataset (no Kaggle account needed):**
```bash
cd data
python generate_data.py           # creates industrial_data.csv
cd ..
```

**Option B — use a real Kaggle dataset:**
1. Download one of the recommended datasets (see spec).
2. Save as `data/industrial_data.csv`.
3. Ensure columns include: `Date`, `Temperature`, `Pressure`, `Speed`, `Torque`, `Downtime_min`, `Production_Output`.

### 3. EDA

```bash
python notebooks/01_eda_analysis.py
# → outputs models/eda_plots.png
```

### 4. Train the Model

```bash
python notebooks/02_model_training.py
# → outputs models/full_model.pkl
# → outputs models/diagnostic_plots.png
# → prints R², RMSE, MAE, UCL/LCL, out-of-control count
```

### 5. Run the Dashboard

```bash
streamlit run dashboard/app.py
# Open http://localhost:8501
```

---

## Dashboard Features

| Tab | Contents |
|-----|----------|
| **Trends & Forecast** | Production trend (actual + predicted + 7-pt rolling avg), 7-hour forecast with confidence bands |
| **Model Performance** | Actual vs predicted scatter, R²/RMSE/MAE metrics, feature coefficient bar chart |
| **Control Chart (SPC)** | Shewhart 3σ chart with OOC markers, residuals vs fitted scatter |
| **Diagnostics** | Q-Q normality plot, residual histogram, IQR outlier table, correlation heatmap |
| **What-if Optimizer** | Interactive parameter sliders, SLSQP constrained optimization, sensitivity sweep |

**Sidebar controls:** date range filter · per-feature alert thresholds · auto-refresh toggle  
**Export buttons:** filtered data CSV · predictions + residuals CSV

---

## Model Details

| Item | Value |
|------|-------|
| Algorithm | `sklearn.linear_model.LinearRegression` |
| Features | Temperature, Pressure, Speed, Torque, Downtime_min + lag1/2/3, rolling_mean_7, interaction terms |
| Scaler | `StandardScaler` (z-score) |
| Split | 80% train / 20% test (time-ordered) |
| Target | `Production_Output` (units/hr) |
| Serialization | `joblib` → `models/full_model.pkl` |

---

## Interpretation Thresholds

| Metric | Target | Action if missed |
|--------|--------|-----------------|
| Test R² | > 0.75 | Add features or try polynomial terms |
| RMSE | < 5% of mean output | Check for outliers or sensor noise |
| Out-of-control points | < 5% | Inspect sensor calibration, raw materials |
| Coefficient sign | Matches domain knowledge | Validate with process engineer |

---

## Deliverables Checklist

- [x] Kaggle-compatible dataset (synthetic or downloaded CSV in `/data`)
- [x] EDA notebook with correlation matrix, outlier analysis, shift boxplot
- [x] Trained linear regression model with R², RMSE, MAE evaluation
- [x] Feature coefficient interpretation (positive/negative bar chart)
- [x] Residual diagnostics: Q-Q plot, residuals vs fitted, histogram
- [x] Interactive dashboard with 5+ interactive widgets (sliders, date picker, tabs, download, threshold controls)
- [x] Production trend line (actual + predicted + rolling avg)
- [x] Actual vs Predicted scatter plot
- [x] Shewhart control chart (3σ) with OOC highlighting and UCL/LCL lines
- [x] Auto-refresh anomaly alert / warning system (sidebar toggle)
- [x] Feature importance / coefficient bar chart
- [x] What-if simulator with SLSQP constrained optimization
- [x] Sensitivity analysis chart
- [x] Export functionality (filtered CSV + predictions CSV)
- [x] README with setup and run instructions

---

## Recommended Kaggle Datasets

| Dataset | Target |
|---------|--------|
| Manufacturing Production Data | Production_Output |
| Predictive Maintenance Dataset | Output / Failure |
| Steel Industry Energy Consumption | Power Consumption |

---

*Industrial Engineering Admin Dashboard — v1.0 · Built for Kaggle-centric manufacturing analytics.*
