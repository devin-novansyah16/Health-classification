"""
predict.py
Load the trained model and predict diabetes risk for new patient data.

Usage:
    python src/predict.py                  # demo with sample patients
    python src/predict.py --interactive    # enter patient data manually
"""

import argparse
import joblib
import numpy as np
import pandas as pd
import sys
import os

# Feature order must match training
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigree", "Age",
    "GlucoseCategory", "BMICategory", "AgeGroup",
]

LABELS = {0: "✅ Healthy (Low Risk)", 1: "⚠️  Diabetic (High Risk)"}


def engineer(d: dict) -> np.ndarray:
    """Apply same feature engineering as training pipeline."""
    g = d["Glucose"]
    b = d["BMI"]
    a = d["Age"]

    d["GlucoseCategory"] = 0 if g <= 100 else (1 if g <= 125 else 2)
    d["BMICategory"]     = 0 if b < 18.5 else (1 if b < 25 else (2 if b < 30 else 3))
    d["AgeGroup"]        = 0 if a <= 30 else (1 if a <= 45 else (2 if a <= 60 else 3))

    return np.array([d[f] for f in FEATURES]).reshape(1, -1)


def predict(patient: dict) -> dict:
    scaler = joblib.load("models/scaler.pkl")
    model  = joblib.load("models/best_model.pkl")

    X = engineer(patient)
    X_scaled = scaler.transform(X)

    pred  = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]

    return {
        "prediction":  pred,
        "label":       LABELS[pred],
        "probability": {
            "Healthy":  round(proba[0] * 100, 1),
            "Diabetic": round(proba[1] * 100, 1),
        }
    }


def print_result(patient: dict, result: dict):
    bar_len = 30
    prob_d  = result["probability"]["Diabetic"] / 100
    filled  = int(prob_d * bar_len)
    bar     = "█" * filled + "░" * (bar_len - filled)

    print("\n" + "─" * 48)
    print("  DIABETES RISK PREDICTION")
    print("─" * 48)
    for k, v in patient.items():
        print(f"  {k:<20}: {v}")
    print("─" * 48)
    print(f"  Prediction : {result['label']}")
    print(f"  Healthy    : {result['probability']['Healthy']:>5}%")
    print(f"  Diabetic   : {result['probability']['Diabetic']:>5}%  [{bar}]")
    print("─" * 48)


def interactive_mode():
    print("\n=== Interactive Diabetes Risk Predictor ===")
    prompts = [
        ("Pregnancies",    "Number of pregnancies",   0,  20),
        ("Glucose",        "Glucose (mg/dL)",          60, 210),
        ("BloodPressure",  "Blood Pressure (mmHg)",    40, 130),
        ("SkinThickness",  "Skin Thickness (mm)",      5,  100),
        ("Insulin",        "Insulin (μU/mL)",          0,  400),
        ("BMI",            "BMI",                      10, 70),
        ("DiabetesPedigree","Diabetes Pedigree (0-2.5)",0, 2.5),
        ("Age",            "Age (years)",              21, 90),
    ]

    patient = {}
    for key, desc, lo, hi in prompts:
        while True:
            try:
                val = float(input(f"  {desc} [{lo}–{hi}]: "))
                if lo <= val <= hi:
                    patient[key] = val
                    break
                print(f"    ↳ Please enter a value between {lo} and {hi}")
            except ValueError:
                print("    ↳ Invalid input, please enter a number.")

    result = predict(patient)
    print_result(patient, result)


# ── Sample demo patients ──────────────────────────────────────────────────────
DEMO_PATIENTS = [
    {
        "name":            "Patient A (Low Risk)",
        "Pregnancies":     1,
        "Glucose":         89,
        "BloodPressure":   66,
        "SkinThickness":   23,
        "Insulin":         94,
        "BMI":             28.1,
        "DiabetesPedigree": 0.167,
        "Age":             21,
    },
    {
        "name":            "Patient B (High Risk)",
        "Pregnancies":     8,
        "Glucose":         183,
        "BloodPressure":   64,
        "SkinThickness":   0,
        "Insulin":         0,
        "BMI":             23.3,
        "DiabetesPedigree": 0.672,
        "Age":             32,
    },
    {
        "name":            "Patient C (Borderline)",
        "Pregnancies":     3,
        "Glucose":         120,
        "BloodPressure":   70,
        "SkinThickness":   28,
        "Insulin":         140,
        "BMI":             31.5,
        "DiabetesPedigree": 0.45,
        "Age":             37,
    },
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true",
                        help="Enter patient data manually")
    args = parser.parse_args()

    if not os.path.exists("models/best_model.pkl"):
        print("[ERROR] Model not found. Please run `python src/train.py` first.")
        sys.exit(1)

    if args.interactive:
        interactive_mode()
    else:
        print("\n=== Demo: Predicting for Sample Patients ===")
        for p in DEMO_PATIENTS:
            patient = {k: v for k, v in p.items() if k != "name"}
            result  = predict(patient)
            patient["_name"] = p["name"]
            print_result(patient, result)
