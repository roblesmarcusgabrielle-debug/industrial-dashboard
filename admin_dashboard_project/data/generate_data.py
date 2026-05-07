"""
Synthetic industrial dataset generator.
Mimics a Kaggle manufacturing dataset:
Temperature, Pressure, Speed, Torque, Downtime -> Production_Output
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 500

dates = pd.date_range(start="2024-01-01", periods=N, freq="h")
shift = ((dates.hour // 8) % 3).map({0: "A", 1: "B", 2: "C"})

temperature  = np.clip(np.random.normal(75, 8, N), 55, 95)
pressure     = np.clip(np.random.normal(100, 12, N), 70, 130)
speed        = np.clip(np.random.normal(1500, 180, N), 1000, 2000)
torque       = np.clip(np.random.normal(35, 8, N), 15, 55)
downtime_min = np.clip(np.random.exponential(5, N), 0, 40)

noise = np.random.normal(0, 20, N)
production_output = (
    300
    + temperature  * 1.2
    + pressure     * 0.8
    + speed        * 0.05
    - torque       * 1.0
    - downtime_min * 3.5
    + noise
)
production_output = np.clip(production_output, 180, 620).round(1)

# Inject 15 anomalous rows
anomaly_idx = np.random.choice(N, 15, replace=False)
production_output[anomaly_idx] += np.random.choice([-80, 90], 15)
production_output = np.clip(production_output, 100, 700).round(1)

df = pd.DataFrame({
    "Date":              dates,
    "Shift":             shift,
    "Temperature":       temperature.round(2),
    "Pressure":          pressure.round(2),
    "Speed":             speed.round(0).astype(int),
    "Torque":            torque.round(2),
    "Downtime_min":      downtime_min.round(2),
    "Production_Output": production_output,
})

df.to_csv("industrial_data.csv", index=False)
print(f"Saved industrial_data.csv — {len(df)} rows")
print(df.describe())
