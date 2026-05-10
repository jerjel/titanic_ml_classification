"""
=======================================================
  TITANIC DATASET - DATA EXPLORATION (EDA)
=======================================================
This script walks through the full EDA pipeline:
  1. Basic data inspection
  2. Missing value analysis
  3. Univariate analysis  (distributions of individual columns)
  4. Bivariate analysis   (feature vs survival rate)
  5. Correlation heatmap
  6. Key summary of findings
=======================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless backend – saves plots to files
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

# ── aesthetics ───────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({
    "figure.facecolor": "#1e1e2e",
    "axes.facecolor":   "#1e1e2e",
    "axes.edgecolor":   "#6e6e8e",
    "axes.labelcolor":  "#cdd6f4",
    "xtick.color":      "#cdd6f4",
    "ytick.color":      "#cdd6f4",
    "text.color":       "#cdd6f4",
    "grid.color":       "#313244",
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
})

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

SURVIVED_COLORS   = ["#f38ba8", "#a6e3a1"]   # red = 0, green = 1
PALETTE_PASTEL    = ["#89b4fa", "#cba6f7", "#f38ba8", "#fab387"]
ACCENT            = "#89b4fa"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 – Load Data
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  TITANIC DATASET – EXPLORATORY DATA ANALYSIS")
print("=" * 60)

df = pd.read_csv("titanic.csv")

print(f"\n► Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print("\n► Column names:\n  ", list(df.columns))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 – Basic Inspection
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  STEP 2 │ BASIC INSPECTION")
print("─" * 60)

print("\n► First 5 rows:")
print(df.head().to_string())

print("\n► Data types & non-null counts:")
print(df.info())

print("\n► Statistical summary:")
print(df.describe().to_string())

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – Missing Value Analysis
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  STEP 3 │ MISSING VALUE ANALYSIS")
print("─" * 60)

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values("Missing %", ascending=False)
print(missing_df.to_string())

# Plot – missing values
fig, ax = plt.subplots(figsize=(8, 4))
fig.patch.set_facecolor("#1e1e2e")
bars = ax.barh(
    missing_df.index,
    missing_df["Missing %"],
    color=PALETTE_PASTEL[:len(missing_df)],
    edgecolor="#313244",
)
for bar, pct in zip(bars, missing_df["Missing %"]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{pct}%", va="center", fontsize=10)
ax.set_xlabel("Missing (%)")
ax.set_title("Missing Values per Column")
ax.set_xlim(0, 110)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/01_missing_values.png", dpi=150)
plt.close()
print(f"\n  ✔ Saved plots/01_missing_values.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 – Univariate Analysis (distributions)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  STEP 4 │ UNIVARIATE ANALYSIS")
print("─" * 60)

# 4a – Survival count
survived_counts = df["Survived"].value_counts()
print(f"\n► Survival counts:\n  Died (0): {survived_counts[0]}  |  Survived (1): {survived_counts[1]}")
print(f"  Survival rate: {survived_counts[1]/len(df)*100:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.patch.set_facecolor("#1e1e2e")
fig.suptitle("Survival Overview", fontsize=14, fontweight="bold")

# Bar chart
axes[0].bar(["Died (0)", "Survived (1)"], survived_counts.values,
            color=SURVIVED_COLORS, edgecolor="#313244", width=0.5)
for i, v in enumerate(survived_counts.values):
    axes[0].text(i, v + 5, str(v), ha="center", fontsize=11)
axes[0].set_title("Count of Passengers")
axes[0].set_ylim(0, max(survived_counts.values) * 1.15)

# Pie chart
axes[1].pie(
    survived_counts.values,
    labels=["Died", "Survived"],
    colors=SURVIVED_COLORS,
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops=dict(edgecolor="#1e1e2e", linewidth=2),
    textprops={"color": "#cdd6f4"},
)
axes[1].set_title("Survival Distribution")

plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/02_survival_overview.png", dpi=150)
plt.close()
print("  ✔ Saved plots/02_survival_overview.png")

# 4b – Age distribution
fig, ax = plt.subplots(figsize=(9, 4))
fig.patch.set_facecolor("#1e1e2e")
ax.hist(df["Age"].dropna(), bins=30, color=ACCENT, edgecolor="#1e1e2e", alpha=0.85)
ax.axvline(df["Age"].median(), color="#f38ba8", linestyle="--", label=f"Median: {df['Age'].median()}")
ax.axvline(df["Age"].mean(),   color="#fab387", linestyle="--", label=f"Mean:   {df['Age'].mean():.1f}")
ax.set_title("Age Distribution")
ax.set_xlabel("Age")
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/03_age_distribution.png", dpi=150)
plt.close()
print("  ✔ Saved plots/03_age_distribution.png")

# 4c – Fare distribution (log-scale for long tail)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.patch.set_facecolor("#1e1e2e")
fig.suptitle("Fare Distribution", fontsize=14, fontweight="bold")
axes[0].hist(df["Fare"], bins=40, color="#cba6f7", edgecolor="#1e1e2e", alpha=0.85)
axes[0].set_title("Raw Fare")
axes[0].set_xlabel("Fare (£)")
axes[1].hist(np.log1p(df["Fare"]), bins=40, color="#cba6f7", edgecolor="#1e1e2e", alpha=0.85)
axes[1].set_title("log(1 + Fare)  – normalised view")
axes[1].set_xlabel("log(1 + Fare)")
for ax in axes:
    ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/04_fare_distribution.png", dpi=150)
plt.close()
print("  ✔ Saved plots/04_fare_distribution.png")

# 4d – Categorical columns
cat_cols = ["Pclass", "Sex", "Embarked", "SibSp", "Parch"]
fig, axes = plt.subplots(1, len(cat_cols), figsize=(16, 4))
fig.patch.set_facecolor("#1e1e2e")
fig.suptitle("Categorical Feature Distributions", fontsize=14, fontweight="bold")
for ax, col in zip(axes, cat_cols):
    counts = df[col].value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values,
           color=PALETTE_PASTEL[:len(counts)], edgecolor="#313244")
    ax.set_title(col)
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 1, str(v), ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/05_categorical_distributions.png", dpi=150)
plt.close()
print("  ✔ Saved plots/05_categorical_distributions.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 – Bivariate Analysis (feature vs survival)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  STEP 5 │ BIVARIATE ANALYSIS (vs Survival)")
print("─" * 60)

def survival_rate_bar(ax, col):
    """Plot survival rate per category as grouped bars."""
    grp = df.groupby(col)["Survived"].value_counts(normalize=True).unstack().fillna(0)
    grp.columns = ["Died", "Survived"]
    x = np.arange(len(grp))
    width = 0.35
    ax.bar(x - width/2, grp["Died"],     width, color="#f38ba8", label="Died",     edgecolor="#313244")
    ax.bar(x + width/2, grp["Survived"], width, color="#a6e3a1", label="Survived", edgecolor="#313244")
    ax.set_xticks(x)
    ax.set_xticklabels(grp.index.astype(str))
    ax.set_ylabel("Proportion")
    ax.set_title(f"Survival Rate by {col}")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)
    # annotate survival rate
    for xi, rate in zip(x, grp["Survived"]):
        ax.text(xi + width/2, rate + 0.02, f"{rate:.0%}", ha="center", fontsize=8)

# 5a – Survival by Pclass, Sex, Embarked
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.patch.set_facecolor("#1e1e2e")
fig.suptitle("Survival Rate by Category", fontsize=14, fontweight="bold")
for ax, col in zip(axes, ["Pclass", "Sex", "Embarked"]):
    survival_rate_bar(ax, col)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/06_survival_by_category.png", dpi=150)
plt.close()
print("  ✔ Saved plots/06_survival_by_category.png")

# Print rates to console
for col in ["Pclass", "Sex", "Embarked"]:
    rates = df.groupby(col)["Survived"].mean().round(3)
    print(f"\n  Survival rate by {col}:\n{rates.to_string()}")

# 5b – Age distribution split by survival
fig, ax = plt.subplots(figsize=(9, 4))
fig.patch.set_facecolor("#1e1e2e")
for val, label, color in [(0, "Died", "#f38ba8"), (1, "Survived", "#a6e3a1")]:
    data = df[df["Survived"] == val]["Age"].dropna()
    ax.hist(data, bins=25, alpha=0.6, label=label, color=color, edgecolor="#1e1e2e")
ax.set_title("Age Distribution: Survived vs Died")
ax.set_xlabel("Age")
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/07_age_vs_survival.png", dpi=150)
plt.close()
print("\n  ✔ Saved plots/07_age_vs_survival.png")

# 5c – Fare vs survival (box plot)
fig, ax = plt.subplots(figsize=(7, 4))
fig.patch.set_facecolor("#1e1e2e")
df_tmp = df.copy()
df_tmp["Survived_Label"] = df_tmp["Survived"].map({0: "Died", 1: "Survived"})
sns.boxplot(
    data=df_tmp, x="Survived_Label", y="Fare",
    palette={"Died": "#f38ba8", "Survived": "#a6e3a1"},
    linewidth=1.5,
    flierprops=dict(marker="o", markerfacecolor="#fab387", markersize=3),
    ax=ax,
)
ax.set_title("Fare Distribution by Survival")
ax.set_xlabel("")
ax.set_ylabel("Fare (£)")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/08_fare_vs_survival.png", dpi=150)
plt.close()
print("  ✔ Saved plots/08_fare_vs_survival.png")

# 5d – Pclass × Sex survival heatmap
pivot = df.groupby(["Pclass", "Sex"])["Survived"].mean().unstack()
fig, ax = plt.subplots(figsize=(6, 4))
fig.patch.set_facecolor("#1e1e2e")
sns.heatmap(
    pivot, annot=True, fmt=".0%",
    cmap="YlGn", linewidths=1, linecolor="#1e1e2e",
    ax=ax, cbar_kws={"shrink": 0.8},
)
ax.set_title("Survival Rate (Pclass × Sex)")
ax.set_xlabel("Sex")
ax.set_ylabel("Passenger Class")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/09_pclass_sex_heatmap.png", dpi=150)
plt.close()
print("  ✔ Saved plots/09_pclass_sex_heatmap.png")

# 5e – Family size feature
df_tmp["FamilySize"] = df_tmp["SibSp"] + df_tmp["Parch"] + 1
fig, ax = plt.subplots(figsize=(9, 4))
fig.patch.set_facecolor("#1e1e2e")
fam_survival = df_tmp.groupby("FamilySize")["Survived"].mean()
ax.bar(fam_survival.index, fam_survival.values, color=ACCENT, edgecolor="#313244")
for x, y in zip(fam_survival.index, fam_survival.values):
    ax.text(x, y + 0.02, f"{y:.0%}", ha="center", fontsize=9)
ax.set_title("Survival Rate by Family Size")
ax.set_xlabel("Family Size (SibSp + Parch + 1)")
ax.set_ylabel("Survival Rate")
ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/10_family_size_vs_survival.png", dpi=150)
plt.close()
print("  ✔ Saved plots/10_family_size_vs_survival.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 – Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  STEP 6 │ CORRELATION HEATMAP")
print("─" * 60)

numeric_df = df[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]].copy()
corr = numeric_df.corr()
print("\n► Correlation with 'Survived':")
print(corr["Survived"].drop("Survived").sort_values(ascending=False).to_string())

fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor("#1e1e2e")
mask = np.triu(np.ones_like(corr, dtype=bool))   # upper triangle mask
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f",
    cmap="coolwarm", center=0,
    linewidths=1, linecolor="#1e1e2e",
    ax=ax, cbar_kws={"shrink": 0.8},
)
ax.set_title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/11_correlation_heatmap.png", dpi=150)
plt.close()
print("  ✔ Saved plots/11_correlation_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  EDA COMPLETE  –  Key Findings")
print("=" * 60)
print(f"""
  Dataset:   {df.shape[0]} passengers, {df.shape[1]} columns
  Survived:  {df['Survived'].sum()} ({df['Survived'].mean()*100:.1f}%)

  Top insights:
  • Women survived at {df[df['Sex']=='female']['Survived'].mean()*100:.0f}% vs men at {df[df['Sex']=='male']['Survived'].mean()*100:.0f}%
  • 1st-class survival: {df[df['Pclass']==1]['Survived'].mean()*100:.0f}%  |  3rd-class: {df[df['Pclass']==3]['Survived'].mean()*100:.0f}%
  • Children (Age < 10): {df[df['Age']<10]['Survived'].mean()*100:.0f}% survival rate
  • Median fare of survivors: £{df[df['Survived']==1]['Fare'].median():.1f} vs £{df[df['Survived']==0]['Fare'].median():.1f} for non-survivors
  • Age: {df['Age'].isnull().sum()} missing ({df['Age'].isnull().mean()*100:.0f}%) → median-fill recommended
  • Cabin: {df['Cabin'].isnull().sum()} missing ({df['Cabin'].isnull().mean()*100:.0f}%) → drop or extract deck letter

  All plots saved to  →  {os.path.abspath(PLOTS_DIR)}/
""")
