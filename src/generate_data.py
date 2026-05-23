"""
generate_data.py
Generates a realistic synthetic dataset for diabetes classification.
Based on clinical features similar to the Pima Indians Diabetes Dataset.
"""

import numpy as np
import pandas as pd

def generate_diabetes_dataset(n_samples: int = 768, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)

    # --- Healthy patients (60%) ---
    n_healthy = int(n_samples * 0.60)
    n_diabetic = n_samples - n_healthy

    healthy = pd.DataFrame({
        "Pregnancies":        np.random.randint(0, 8, n_healthy),
        "Glucose":            np.random.normal(90, 15, n_healthy).clip(60, 140).astype(int),
        "BloodPressure":      np.random.normal(70, 10, n_healthy).clip(50, 95).astype(int),
        "SkinThickness":      np.random.normal(22, 8, n_healthy).clip(5, 45).astype(int),
        "Insulin":            np.random.normal(80, 40, n_healthy).clip(0, 200).astype(int),
        "BMI":                np.round(np.random.normal(27, 5, n_healthy).clip(18, 40), 1),
        "DiabetesPedigree":   np.round(np.random.uniform(0.05, 0.8, n_healthy), 3),
        "Age":                np.random.randint(21, 55, n_healthy),
        "Outcome":            np.zeros(n_healthy, dtype=int),
    })

    # --- Diabetic patients (40%) ---
    diabetic = pd.DataFrame({
        "Pregnancies":        np.random.randint(0, 15, n_diabetic),
        "Glucose":            np.random.normal(145, 25, n_diabetic).clip(100, 210).astype(int),
        "BloodPressure":      np.random.normal(78, 12, n_diabetic).clip(50, 110).astype(int),
        "SkinThickness":      np.random.normal(32, 10, n_diabetic).clip(10, 60).astype(int),
        "Insulin":            np.random.normal(170, 80, n_diabetic).clip(0, 400).astype(int),
        "BMI":                np.round(np.random.normal(35, 6, n_diabetic).clip(22, 55), 1),
        "DiabetesPedigree":   np.round(np.random.uniform(0.2, 2.0, n_diabetic), 3),
        "Age":                np.random.randint(30, 75, n_diabetic),
        "Outcome":            np.ones(n_diabetic, dtype=int),
    })

    df = pd.concat([healthy, diabetic], ignore_index=True).sample(frac=1, random_state=random_state)

    # Inject realistic missing values (zeros) like the original dataset
    for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        mask = np.random.choice([True, False], size=len(df), p=[0.04, 0.96])
        df.loc[mask, col] = 0

    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = generate_diabetes_dataset()
    df.to_csv("data/diabetes.csv", index=False)
    print(f"Dataset saved: {df.shape[0]} rows x {df.shape[1]} cols")
    print(df["Outcome"].value_counts().rename({0: "Healthy", 1: "Diabetic"}))
