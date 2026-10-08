# -*- coding: utf-8 -*-
"""
Weather Prediction Model
------------------------
Reads weather_24_years_IBM_Bob.csv, reshapes from wide to long format,
trains simple Linear Regression models for Temperature and Rainfall,
then lets you predict for any month and year interactively.
"""
import sys, io
# Force UTF-8 output so the script works on any terminal
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# ── 1. Load the CSV ──────────────────────────────────────────────────────────
df = pd.read_csv("weather_24_years_IBM_Bob.csv")

# ── 2. Reshape wide → long ───────────────────────────────────────────────────
# Month abbreviations that appear in the column names
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_NUM = {m: i + 1 for i, m in enumerate(MONTHS)}  # Jan→1, Feb→2 …

rows = []
for _, row in df.iterrows():
    year = int(row["Year"])
    for month_abbr in MONTHS:
        temp_col = f"{month_abbr}_Temperature_C"
        rain_col = f"{month_abbr}_Rainfall"
        if temp_col in df.columns and rain_col in df.columns:
            rows.append({
                "Year":        year,
                "Month":       MONTH_NUM[month_abbr],
                "Temperature": row[temp_col],
                "Rainfall":    row[rain_col],
            })

long_df = pd.DataFrame(rows)
print(f"Dataset reshaped: {len(long_df)} rows  (years × months)")
print(long_df.head(6).to_string(index=False))
print()

# ── 3. Feature engineering ───────────────────────────────────────────────────
# Features: Year, Month, sin/cos of month (captures seasonality)
def make_features(year, month):
    """Return a feature array for a given year and month."""
    angle = 2 * np.pi * month / 12
    return [[year, month, np.sin(angle), np.cos(angle)]]

X = np.array([
    make_features(r["Year"], r["Month"])[0]
    for _, r in long_df.iterrows()
])
y_temp = long_df["Temperature"].values
y_rain = long_df["Rainfall"].values

# ── 4. Train models ──────────────────────────────────────────────────────────
temp_model = LinearRegression()
temp_model.fit(X, y_temp)

rain_model = LinearRegression()
rain_model.fit(X, y_rain)

# Quick training-set accuracy report
temp_preds = temp_model.predict(X)
rain_preds = rain_model.predict(X)

print("=== Model Training Summary ===")
print(f"Temperature  MAE : {mean_absolute_error(y_temp, temp_preds):.2f} °C")
print(f"Rainfall     MAE : {mean_absolute_error(y_rain, rain_preds):.0f} mm")
print()

# ── 5. Interactive prediction ─────────────────────────────────────────────────
MONTH_NAMES = {
    1: "January",  2: "February",  3: "March",     4: "April",
    5: "May",      6: "June",      7: "July",       8: "August",
    9: "September",10: "October",  11: "November", 12: "December",
}

print("=== Weather Predictor ===")
print("Enter any year and month to get a temperature & rainfall prediction.")
print("Type 'quit' to exit.\n")

while True:
    try:
        year_input = input("Enter year  (e.g. 2025): ").strip()
        if year_input.lower() == "quit":
            break

        month_input = input("Enter month (1–12)     : ").strip()
        if month_input.lower() == "quit":
            break

        year  = int(year_input)
        month = int(month_input)

        if not (1 <= month <= 12):
            print("  ⚠  Month must be between 1 and 12. Try again.\n")
            continue

        features = make_features(year, month)
        pred_temp = temp_model.predict(features)[0]
        pred_rain = rain_model.predict(features)[0]
        pred_rain = max(0, pred_rain)   # rainfall can't be negative

        print(f"\n  >>  {MONTH_NAMES[month]} {year}")
        print(f"  Temp  : {pred_temp:.1f} deg C")
        print(f"  Rain  : {pred_rain:.0f} mm\n")

    except ValueError:
        print("  [!] Please enter valid numbers.\n")
    except KeyboardInterrupt:
        print("\nExiting.")
        break

print("Done.")
