"""
02_model_training.py
Train LinearRegression on the industrial dataset.
Saves models/full_model.pkl containing model, scaler, features, metrics.
Run: python notebooks/02_model_training.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import scipy.stats as stats

# ── 1. Load ──────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "industrial_data.csv")
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
print(f"Loaded {len(df)} rows.")
print(df.info())
print(df.describe())

# ── 2. Outlier detection (IQR) ───────────────────────────────────────────────
print("\n── IQR Outlier Detection ──")
for col in ["Temperature", "Pressure", "Speed", "Torque", "Downtime_min", "Production_Output"]:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    out = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    print(f"  {col}: {len(out)} outliers")

# ── 3. Feature engineering ───────────────────────────────────────────────────
df = df.sort_values("Date").reset_index(drop=True)
df["output_lag1"]      = df["Production_Output"].shift(1)
df["output_lag2"]      = df["Production_Output"].shift(2)
df["output_lag3"]      = df["Production_Output"].shift(3)
df["rolling_mean_7"]   = df["Production_Output"].rolling(7).mean()
df["temp_x_speed"]     = df["Temperature"] * df["Speed"] / 1000   # interaction
df["pressure_x_torque"]= df["Pressure"]    * df["Torque"] / 100
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

FEATURES = [
    "Temperature", "Pressure", "Speed", "Torque", "Downtime_min",
    "output_lag1", "output_lag2", "output_lag3",
    "rolling_mean_7", "temp_x_speed", "pressure_x_torque",
]
TARGET = "Production_Output"

X = df[FEATURES]
y = df[TARGET]

# ── 4. Scale + split ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False   # time-aware split
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
X_all_s   = scaler.transform(X)

# ── 5. Train ─────────────────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train_s, y_train)

preds      = model.predict(X_test_s)
preds_all  = model.predict(X_all_s)

r2   = r2_score(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
mae  = mean_absolute_error(y_test, preds)

print(f"\n── Model Metrics ──")
print(f"  Test R²  : {r2:.4f}")
print(f"  RMSE     : {rmse:.2f} units")
print(f"  MAE      : {mae:.2f} units")
print(f"  RMSE %   : {rmse/y_test.mean()*100:.1f}% of mean output")

# ── 6. Residual / control chart stats ───────────────────────────────────────
residuals  = y_test.values - preds
mean_res   = residuals.mean()
std_res    = residuals.std()
UCL        = mean_res + 3 * std_res
LCL        = mean_res - 3 * std_res
ooc_mask   = (residuals > UCL) | (residuals < LCL)
ooc_count  = ooc_mask.sum()
ooc_pct    = ooc_count / len(residuals) * 100

print(f"\n── Control Chart ──")
print(f"  Mean residual : {mean_res:.3f}")
print(f"  Std  residual : {std_res:.3f}")
print(f"  UCL           : {UCL:.3f}")
print(f"  LCL           : {LCL:.3f}")
print(f"  Out-of-control: {ooc_count} ({ooc_pct:.1f}%)")

if ooc_pct < 5:
    print("  ✓ Process STABLE — OOC < 5%")
else:
    print("  ⚠ Process UNSTABLE — investigate sensor drift or material change")

# ── 7. Coefficient interpretation ────────────────────────────────────────────
coef_df = pd.DataFrame({
    "Feature":     FEATURES,
    "Coefficient": model.coef_,
}).sort_values("Coefficient", key=abs, ascending=False)
print("\n── Feature Coefficients (scaled) ──")
print(coef_df.to_string(index=False))

# ── 8. Save model package ────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

model_package = {
    "model":     model,
    "scaler":    scaler,
    "features":  FEATURES,
    "metrics": {
        "r2":       round(r2,   4),
        "rmse":     round(rmse, 2),
        "mae":      round(mae,  2),
        "UCL":      round(UCL,  4),
        "LCL":      round(LCL,  4),
        "mean_res": round(mean_res, 4),
        "std_res":  round(std_res,  4),
        "ooc_count": int(ooc_count),
        "ooc_pct":  round(ooc_pct, 2),
    },
    "coef_df":    coef_df,
    "df":         df,
    "X_test":     X_test,
    "y_test":     y_test.values,
    "preds":      preds,
    "preds_all":  preds_all,
    "residuals":  residuals,
}
out_path = os.path.join(MODELS_DIR, "full_model.pkl")
joblib.dump(model_package, out_path)
print(f"\n✓ Model saved → {out_path}")

# ── 9. Diagnostic plots ───────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Industrial LR — Diagnostic Plots", fontsize=13, y=0.98)

# Actual vs Predicted
ax = axes[0, 0]
ax.scatter(y_test, preds, alpha=0.5, s=20, color="#185FA5")
mn, mx = min(y_test.min(), preds.min()), max(y_test.max(), preds.max())
ax.plot([mn, mx], [mn, mx], "r--", lw=1)
ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
ax.set_title(f"Actual vs Predicted  (R²={r2:.3f})")

# Residuals vs Fitted
ax = axes[0, 1]
ax.scatter(preds, residuals, alpha=0.5, s=20, color="#1D9E75")
ax.axhline(0, color="red", lw=1, ls="--")
ax.axhline(UCL, color="orange", lw=1, ls="--", label="UCL/LCL")
ax.axhline(LCL, color="orange", lw=1, ls="--")
ax.scatter(preds[ooc_mask], residuals[ooc_mask], color="red", s=40, zorder=5, label="OOC")
ax.set_xlabel("Fitted"); ax.set_ylabel("Residual")
ax.set_title("Residuals vs Fitted"); ax.legend(fontsize=8)

# Q-Q plot
ax = axes[1, 0]
(osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
ax.scatter(osm, osr, alpha=0.5, s=20, color="#185FA5")
ax.plot(osm, slope*np.array(osm)+intercept, "r--", lw=1)
ax.set_xlabel("Theoretical quantiles"); ax.set_ylabel("Sample quantiles")
ax.set_title("Q-Q Plot (Normality Check)")

# Residual histogram
ax = axes[1, 1]
ax.hist(residuals, bins=20, color="#185FA5", alpha=0.7, edgecolor="white")
ax.axvline(UCL, color="red", lw=1, ls="--", label="UCL")
ax.axvline(LCL, color="red", lw=1, ls="--", label="LCL")
ax.set_xlabel("Residual"); ax.set_ylabel("Count")
ax.set_title("Residual Distribution"); ax.legend(fontsize=8)

plt.tight_layout()
plot_path = os.path.join(MODELS_DIR, "diagnostic_plots.png")
plt.savefig(plot_path, dpi=120, bbox_inches="tight")
print(f"✓ Diagnostic plots saved → {plot_path}")
