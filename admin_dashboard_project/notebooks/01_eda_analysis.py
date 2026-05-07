"""
01_eda_analysis.py  (mirrors a Jupyter notebook in script form)
Exploratory Data Analysis on the industrial manufacturing dataset.
Run: python notebooks/01_eda_analysis.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── 1. Load ──────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "industrial_data.csv")
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
print("── Shape ──"); print(df.shape)
print("\n── dtypes ──"); print(df.dtypes)
print("\n── Missing values ──"); print(df.isnull().sum())
print("\n── Describe ──"); print(df.describe())

# ── 2. Outlier detection (IQR) ───────────────────────────────────────────────
print("\n── IQR Outlier Report ──")
num_cols = ["Temperature", "Pressure", "Speed", "Torque", "Downtime_min", "Production_Output"]
outlier_report = []
for col in num_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    mask = (df[col] < lo) | (df[col] > hi)
    outlier_report.append({"Feature": col, "Q1": round(Q1,2), "Q3": round(Q3,2),
                            "IQR": round(IQR,2), "Lower fence": round(lo,2),
                            "Upper fence": round(hi,2), "Outliers": mask.sum()})
print(pd.DataFrame(outlier_report).to_string(index=False))

# ── 3. Correlation matrix ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("EDA — Industrial Manufacturing Dataset", fontsize=14, y=1.01)

corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            ax=axes[0, 0], linewidths=0.3, cbar_kws={"shrink": 0.8})
axes[0, 0].set_title("Correlation Matrix")

# Target distribution
axes[0, 1].hist(df["Production_Output"], bins=30, color="#185FA5", alpha=0.75, edgecolor="white")
axes[0, 1].set_title("Production Output Distribution")
axes[0, 1].set_xlabel("Units/hr")

# Production over time
axes[0, 2].plot(df["Date"], df["Production_Output"], lw=0.7, color="#1D9E75", alpha=0.8)
axes[0, 2].plot(df["Date"],
                df["Production_Output"].rolling(24).mean(),
                lw=1.5, color="#185FA5", label="24h rolling avg")
axes[0, 2].set_title("Production Output Over Time")
axes[0, 2].legend(fontsize=8)
axes[0, 2].tick_params(axis="x", rotation=30)

# Scatter: Temperature vs Output
axes[1, 0].scatter(df["Temperature"], df["Production_Output"], alpha=0.3, s=10, color="#185FA5")
axes[1, 0].set_xlabel("Temperature (°C)"); axes[1, 0].set_ylabel("Production Output")
axes[1, 0].set_title("Temperature vs Output")

# Scatter: Downtime vs Output
axes[1, 1].scatter(df["Downtime_min"], df["Production_Output"], alpha=0.3, s=10, color="#BA7517")
axes[1, 1].set_xlabel("Downtime (min)"); axes[1, 1].set_ylabel("Production Output")
axes[1, 1].set_title("Downtime vs Output")

# Boxplot by Shift
shifts = df.groupby("Shift")["Production_Output"].apply(list)
axes[1, 2].boxplot([shifts["A"], shifts["B"], shifts["C"]],
                   labels=["Shift A", "Shift B", "Shift C"],
                   patch_artist=True,
                   boxprops=dict(facecolor="#E6F1FB"),
                   medianprops=dict(color="#185FA5", lw=2))
axes[1, 2].set_title("Output by Shift")

plt.tight_layout()
out_dir = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, "eda_plots.png"), dpi=120, bbox_inches="tight")
print("\n✓ EDA plots saved → models/eda_plots.png")

# ── 4. Time-based feature preview ────────────────────────────────────────────
df_feat = df.sort_values("Date").copy()
df_feat["lag1"]        = df_feat["Production_Output"].shift(1)
df_feat["rolling7"]    = df_feat["Production_Output"].rolling(7).mean()
print("\n── Feature-engineered preview (first 5 non-null rows) ──")
print(df_feat[["Date", "Production_Output", "lag1", "rolling7"]].dropna().head())
