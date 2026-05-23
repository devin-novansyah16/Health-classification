"""
preprocess.py
Data cleaning and preprocessing pipeline for diabetes classification.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os


ZERO_AS_NAN_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
FEATURE_COLS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigree", "Age"]
TARGET_COL = "Outcome"


def load_data(path: str = "data/diabetes.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[INFO] Loaded dataset: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Replace biologically impossible zeros with NaN, then impute with median."""
    df = df.copy()
    for col in ZERO_AS_NAN_COLS:
        df[col] = df[col].replace(0, np.nan)

    missing = df.isnull().sum()
    if missing.any():
        print("[INFO] Missing values detected — imputing with median:")
        print(missing[missing > 0])
        df.fillna(df.median(numeric_only=True), inplace=True)

    return df


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-informed features."""
    df = df.copy()
    df["GlucoseCategory"] = pd.cut(
        df["Glucose"],
        bins=[0, 100, 125, 200],
        labels=[0, 1, 2]          # normal, pre-diabetic, diabetic
    ).astype(float)

    df["BMICategory"] = pd.cut(
        df["BMI"],
        bins=[0, 18.5, 25, 30, 100],
        labels=[0, 1, 2, 3]       # underweight, normal, overweight, obese
    ).astype(float)

    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 30, 45, 60, 100],
        labels=[0, 1, 2, 3]
    ).astype(float)

    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    base = FEATURE_COLS.copy()
    extra = ["GlucoseCategory", "BMICategory", "AgeGroup"]
    return base + [c for c in extra if c in df.columns]


def split_and_scale(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Return train/test splits (scaled X, raw y) and the fitted scaler."""
    features = get_feature_cols(df)
    df = df.copy()
    for col in features:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    X = df[features].values
    y = df[TARGET_COL].values

    # Ensure no NaN leaks (engineered categoricals may produce NaN at edges)
    df_filled = df.copy()
    for col in get_feature_cols(df):
        if col in df_filled.columns:
            df_filled[col] = df_filled[col].fillna(df_filled[col].median())
    X = df_filled[features].values
    y = df_filled[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    print("[INFO] Scaler saved → models/scaler.pkl")

    return X_train_sc, X_test_sc, y_train, y_test, features


def run_pipeline(path: str = "data/diabetes.csv"):
    df = load_data(path)
    df = clean_data(df)
    df = feature_engineer(df)
    return split_and_scale(df)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features = run_pipeline()
    print(f"\n[RESULT] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"[RESULT] Features used: {features}")
