"""
train.py
Trains multiple ML classifiers, tunes the best one, and saves the final model.
Models compared: Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

import sys
sys.path.insert(0, "src")
from preprocess import run_pipeline

# ── Paths ────────────────────────────────────────────────────────────────────
os.makedirs("models",  exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test, features = run_pipeline()

# ── Define candidate models ───────────────────────────────────────────────────
MODELS = {
    "Logistic Regression":    LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":          RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":      GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM":                    SVC(probability=True, random_state=42),
    "K-Nearest Neighbors":    KNeighborsClassifier(n_neighbors=7),
}

# ── Cross-validation comparison ───────────────────────────────────────────────
print("\n===== Cross-Validation Comparison (5-fold) =====")
cv_results = {}
for name, model in MODELS.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")
    cv_results[name] = {"mean": scores.mean(), "std": scores.std()}
    print(f"  {name:<25} F1 = {scores.mean():.4f} ± {scores.std():.4f}")

best_name = max(cv_results, key=lambda k: cv_results[k]["mean"])
print(f"\n[INFO] Best model by CV F1: {best_name}")

# ── Hyperparameter tuning on best model ───────────────────────────────────────
print("\n===== Hyperparameter Tuning =====")
param_grids = {
    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth":    [None, 5, 10],
        "min_samples_split": [2, 5],
    },
    "Gradient Boosting": {
        "n_estimators":  [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth":     [3, 5],
    },
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["lbfgs", "liblinear"],
    },
    "SVM": {
        "C":      [0.1, 1, 10],
        "kernel": ["rbf", "linear"],
    },
    "K-Nearest Neighbors": {
        "n_neighbors": [3, 5, 7, 11],
        "weights":     ["uniform", "distance"],
    },
}

base_model  = MODELS[best_name]
param_grid  = param_grids.get(best_name, {})

if param_grid:
    grid_search = GridSearchCV(
        base_model, param_grid, cv=5, scoring="f1",
        n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)
    final_model = grid_search.best_estimator_
    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Best CV F1:  {grid_search.best_score_:.4f}")
else:
    final_model = base_model
    final_model.fit(X_train, y_train)

final_model.fit(X_train, y_train)

# ── Evaluate on test set ──────────────────────────────────────────────────────
y_pred  = final_model.predict(X_test)
y_proba = final_model.predict_proba(X_test)[:, 1]

metrics = {
    "model":     best_name,
    "accuracy":  round(accuracy_score(y_test, y_pred),  4),
    "precision": round(precision_score(y_test, y_pred), 4),
    "recall":    round(recall_score(y_test, y_pred),    4),
    "f1":        round(f1_score(y_test, y_pred),        4),
    "roc_auc":   round(roc_auc_score(y_test, y_proba),  4),
}

print("\n===== Test-Set Evaluation =====")
for k, v in metrics.items():
    print(f"  {k:<12}: {v}")

print("\n" + classification_report(y_test, y_pred, target_names=["Healthy", "Diabetic"]))

# ── Save model & metrics ──────────────────────────────────────────────────────
joblib.dump(final_model, "models/best_model.pkl")
with open("reports/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("[INFO] Model saved → models/best_model.pkl")
print("[INFO] Metrics saved → reports/metrics.json")

# ── Plot 1: CV F1 Comparison ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor("#0f172a")
for ax in axes:
    ax.set_facecolor("#1e293b")

names  = list(cv_results.keys())
means  = [cv_results[n]["mean"] for n in names]
stds   = [cv_results[n]["std"]  for n in names]
colors = ["#38bdf8" if n != best_name else "#f97316" for n in names]

bars = axes[0].barh(names, means, xerr=stds, color=colors,
                    error_kw={"ecolor": "#94a3b8", "capsize": 4}, height=0.6)
axes[0].set_xlabel("F1 Score (CV)", color="#e2e8f0")
axes[0].set_title("Model Comparison", color="#f8fafc", fontsize=13, fontweight="bold")
axes[0].tick_params(colors="#e2e8f0")
for spine in axes[0].spines.values():
    spine.set_edgecolor("#334155")
axes[0].set_xlim(0, 1)
axes[0].axvline(0.5, color="#475569", linewidth=0.8, linestyle="--")

# ── Plot 2: Confusion Matrix ──────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
            xticklabels=["Healthy", "Diabetic"],
            yticklabels=["Healthy", "Diabetic"],
            ax=axes[1], cbar=False,
            annot_kws={"size": 14, "weight": "bold"})
axes[1].set_title("Confusion Matrix", color="#f8fafc", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Predicted", color="#e2e8f0")
axes[1].set_ylabel("Actual", color="#e2e8f0")
axes[1].tick_params(colors="#e2e8f0")

# ── Plot 3: Feature Importance ────────────────────────────────────────────────
if hasattr(final_model, "feature_importances_"):
    importances = final_model.feature_importances_
    sorted_idx  = np.argsort(importances)
    feat_names  = [features[i] for i in sorted_idx]
    feat_vals   = importances[sorted_idx]
    axes[2].barh(feat_names, feat_vals, color="#38bdf8", height=0.6)
    axes[2].set_title("Feature Importance", color="#f8fafc", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Importance", color="#e2e8f0")
    axes[2].tick_params(colors="#e2e8f0")
    for spine in axes[2].spines.values():
        spine.set_edgecolor("#334155")
elif hasattr(final_model, "coef_"):
    coef = np.abs(final_model.coef_[0])
    sorted_idx = np.argsort(coef)
    feat_names = [features[i] for i in sorted_idx]
    axes[2].barh(feat_names, coef[sorted_idx], color="#38bdf8", height=0.6)
    axes[2].set_title("Feature Coefficients (abs)", color="#f8fafc", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("|Coefficient|", color="#e2e8f0")
    axes[2].tick_params(colors="#e2e8f0")
    for spine in axes[2].spines.values():
        spine.set_edgecolor("#334155")

plt.tight_layout()
plt.savefig("reports/model_evaluation.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("[INFO] Plot saved → reports/model_evaluation.png")
print("\n[DONE] Training complete!")
