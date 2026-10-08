# -*- coding: utf-8 -*-
"""
Weather Prediction Web App
--------------------------
Run:  python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# ── Load & reshape data ──────────────────────────────────────────────────────
df = pd.read_csv("weather_24_years_IBM_Bob.csv")

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_NUM = {m: i + 1 for i, m in enumerate(MONTHS)}

rows = []
for _, row in df.iterrows():
    year = int(row["Year"])
    for abbr in MONTHS:
        tc, rc = f"{abbr}_Temperature_C", f"{abbr}_Rainfall"
        if tc in df.columns and rc in df.columns:
            rows.append({
                "Year": year, "Month": MONTH_NUM[abbr],
                "Temperature": row[tc], "Rainfall": row[rc],
            })

long_df = pd.DataFrame(rows)

# ── Feature engineering ──────────────────────────────────────────────────────
def make_features(year, month):
    angle = 2 * np.pi * month / 12
    return [[year, month, np.sin(angle), np.cos(angle)]]

X = np.array([make_features(r["Year"], r["Month"])[0]
              for _, r in long_df.iterrows()])
y_temp = long_df["Temperature"].values
y_rain = long_df["Rainfall"].values

# ── Train models ─────────────────────────────────────────────────────────────
temp_model = LinearRegression().fit(X, y_temp)
rain_model = LinearRegression().fit(X, y_rain)

print("Models trained. Open http://127.0.0.1:5000 in your browser.")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        year  = int(data["year"])
        month = int(data["month"])
        if not (1 <= month <= 12):
            return jsonify({"error": "Month must be between 1 and 12."}), 400
    except (KeyError, ValueError):
        return jsonify({"error": "Invalid input."}), 400

    features  = make_features(year, month)
    pred_temp = round(float(temp_model.predict(features)[0]), 1)
    pred_rain = round(max(0.0, float(rain_model.predict(features)[0])), 0)

    return jsonify({"temperature": pred_temp, "rainfall": pred_rain})


if __name__ == "__main__":
    app.run(debug=True)
