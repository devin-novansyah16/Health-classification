"""
eda.py
Exploratory Data Analysis for the Diabetes Classification dataset.
Generates a multi-panel report saved to reports/eda_report.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("reports", exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/diabetes.csv")
df_clean = df.copy()
for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
    df_clean[col] = df_clean[col].replace(0, np.nan)
df_clean.fillna(df_clean.median(numeric_only=True), inplace=True)

feature_cols = ["Pregnancies", "Glucose", "BloodPressure",
                "SkinThickness", "Insulin", "BMI", "DiabetesPedigree", "Age"]

BG   = "#0f172a"
CARD = "#1e293b"
LINE = "#334155"
TXT  = "#e2e8f0"
BLUE = "#38bdf8"
ORG  = "#f97316"
GRN  = "#4ade80"

# ── Figure setup ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 22), facecolor=BG)
fig.suptitle("Diabetes Dataset — Exploratory Data Analysis",
             color="#f8fafc", fontsize=20, fontweight="bold", y=0.98)

gs = fig.add_gridspec(4, 4, hspace=0.45, wspace=0.4)

def style(ax):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=TXT, labelsize=8)
    ax.xaxis.label.set_color(TXT)
    ax.yaxis.label.set_color(TXT)
    ax.title.set_color("#f8fafc")
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)

# 1. Class distribution (pie) ─────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
style(ax1)
counts = df["Outcome"].value_counts()
ax1.pie(counts, labels=["Healthy", "Diabetic"], colors=[BLUE, ORG],
        autopct="%1.1f%%", startangle=90,
        textprops={"color": TXT, "fontsize": 10},
        wedgeprops={"edgecolor": BG, "linewidth": 2})
ax1.set_title("Class Distribution")

# 2. Missing values bar ───────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
style(ax2)
raw_zeros = {c: (df[c] == 0).sum() for c in
             ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]}
ax2.barh(list(raw_zeros.keys()), list(raw_zeros.values()), color=ORG, height=0.6)
ax2.set_title("Zero / Missing Values")
ax2.set_xlabel("Count")

# 3. Glucose distribution by class ────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2:])
style(ax3)
for outcome, color, label in [(0, BLUE, "Healthy"), (1, ORG, "Diabetic")]:
    vals = df_clean[df_clean["Outcome"] == outcome]["Glucose"]
    ax3.hist(vals, bins=30, alpha=0.7, color=color, label=label, edgecolor=BG)
ax3.set_title("Glucose Distribution by Class")
ax3.set_xlabel("Glucose (mg/dL)")
ax3.set_ylabel("Frequency")
ax3.legend(facecolor=CARD, labelcolor=TXT)

# 4–11. Feature distributions (boxplots per feature) ─────────────────────────
for i, col in enumerate(feature_cols):
    row = (i // 4) + 1
    col_idx = i % 4
    ax = fig.add_subplot(gs[row, col_idx])
    style(ax)
    data0 = df_clean[df_clean["Outcome"] == 0][col]
    data1 = df_clean[df_clean["Outcome"] == 1][col]
    bp = ax.boxplot([data0, data1], patch_artist=True,
                    medianprops={"color": "#f8fafc", "linewidth": 2},
                    whiskerprops={"color": LINE},
                    capprops={"color": LINE},
                    flierprops={"marker": "o", "markerfacecolor": LINE,
                                "markersize": 3, "alpha": 0.5})
    bp["boxes"][0].set_facecolor(BLUE + "99")
    bp["boxes"][1].set_facecolor(ORG + "99")
    ax.set_title(col, fontsize=9)
    ax.set_xticklabels(["Healthy", "Diabetic"], fontsize=8)

# 12. Correlation heatmap ─────────────────────────────────────────────────────
ax_corr = fig.add_subplot(gs[3, :])
style(ax_corr)
corr = df_clean[feature_cols + ["Outcome"]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(corr, ax=ax_corr, mask=mask, cmap=cmap, center=0,
            annot=True, fmt=".2f", annot_kws={"size": 8},
            linewidths=0.5, linecolor=BG,
            cbar_kws={"shrink": 0.8})
ax_corr.set_title("Feature Correlation Matrix", fontsize=12)
ax_corr.tick_params(labelsize=8)

plt.savefig("reports/eda_report.png", dpi=150, bbox_inches="tight",
            facecolor=BG)
plt.close()

# ── Print summary ─────────────────────────────────────────────────────────────
print("===== Dataset Summary =====")
print(f"  Rows    : {df.shape[0]}")
print(f"  Columns : {df.shape[1]}")
print(f"\n  Class balance:")
print(df["Outcome"].value_counts(normalize=True).rename({0:"Healthy",1:"Diabetic"}).map("{:.1%}".format))
print(f"\n  Descriptive statistics:")
print(df_clean[feature_cols].describe().round(2).to_string())
print("\n[INFO] EDA report saved → reports/eda_report.png")
