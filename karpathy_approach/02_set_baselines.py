"""
╔══════════════════════════════════════════════════════════════╗
║  STEP 2 — SET A DUMB BASELINE                               ║
║  Karpathy's Rule #2: "Always establish a simple baseline    ║
║  before building anything complex. It tells you the floor   ║
║  of your performance."                                       ║
╚══════════════════════════════════════════════════════════════╝

WHY THIS STEP?
  A "dumb" baseline gives you the minimum bar you must beat.
  It also validates the entire evaluation pipeline (train/test
  split, metrics, logging) before adding model complexity.

  Three baselines in increasing sophistication:
    B0 – Majority-class predictor  (everyone dies → 61.6% acc)
    B1 – Gender rule               (women survive, men die)
    B2 – Logistic Regression on 2 raw features (Pclass + Sex)

  If your fancy model cannot beat B1, something is very wrong.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler
import os, json

# ── paths ────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "titanic.csv")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
LOG_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_log.md")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── aesthetics ───────────────────────────────────────────────────────────────
BG, FG  = "#0f0f1a", "#cdd6f4"
GRID    = "#1e1e2e"
PALETTE = ["#89b4fa", "#cba6f7", "#a6e3a1", "#f38ba8", "#fab387"]
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": GRID,
    "axes.edgecolor": "#6e6e8e", "axes.labelcolor": FG,
    "xtick.color": FG, "ytick.color": FG, "text.color": FG,
    "grid.color": "#313244",
})

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*65)
print("  STEP 2 │ DUMB BASELINES")
print("═"*65)

# ── load & minimal prep ──────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df["Embarked"].fillna("S", inplace=True)
df["Age"].fillna(df["Age"].median(), inplace=True)

# Encode Sex (binary)
df["Sex_enc"] = (df["Sex"] == "female").astype(int)

# Fixed 80/20 split — same random_state used across ALL steps for fair comparison
X_all = df[["Pclass","Sex_enc","Age","Fare","SibSp","Parch"]]
y_all = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

results = {}

# ─────────────────────────────────────────────────────────────────────────────
# B0 – Majority-class predictor
# ─────────────────────────────────────────────────────────────────────────────
print("\n  [B0] Majority-class predictor (everyone dies)")
majority_class = y_train.mode()[0]
b0_pred = np.full(len(y_test), majority_class)
b0_acc  = accuracy_score(y_test, b0_pred)
results["B0_majority"] = round(b0_acc, 4)
print(f"      Accuracy: {b0_acc*100:.2f}%")
print(f"      This is your floor — any model MUST beat this.\n")

# ─────────────────────────────────────────────────────────────────────────────
# B1 – Gender rule  (domain knowledge: "women and children first")
# ─────────────────────────────────────────────────────────────────────────────
print("  [B1] Gender rule: female → survived, male → died")
b1_pred = (X_test["Sex_enc"] == 1).astype(int).values
b1_acc  = accuracy_score(y_test, b1_pred)
results["B1_gender"] = round(b1_acc, 4)
print(f"      Accuracy: {b1_acc*100:.2f}%")
print(f"      Insight: A single feature (Sex) captures most of the signal!")
print(classification_report(y_test, b1_pred, target_names=["Died","Survived"]))

# ─────────────────────────────────────────────────────────────────────────────
# B2 – Logistic Regression on 2 features (Pclass + Sex_enc)
# ─────────────────────────────────────────────────────────────────────────────
print("  [B2] Logistic Regression on {Pclass, Sex} only")
X_b2_train = X_train[["Pclass","Sex_enc"]]
X_b2_test  = X_test[["Pclass","Sex_enc"]]
scaler = StandardScaler()
X_b2_train_s = scaler.fit_transform(X_b2_train)
X_b2_test_s  = scaler.transform(X_b2_test)

lr = LogisticRegression(max_iter=200, random_state=42)
lr.fit(X_b2_train_s, y_train)
b2_pred = lr.predict(X_b2_test_s)
b2_acc  = accuracy_score(y_test, b2_pred)
b2_cv   = cross_val_score(lr, scaler.fit_transform(X_all[["Pclass","Sex_enc"]]),
                          y_all, cv=5).mean()
results["B2_logreg_2feat"] = round(b2_acc, 4)
print(f"      Test accuracy : {b2_acc*100:.2f}%")
print(f"      5-fold CV mean: {b2_cv*100:.2f}%")
print(classification_report(y_test, b2_pred, target_names=["Died","Survived"]))

# ─────────────────────────────────────────────────────────────────────────────
# Plot: Baseline comparison bar chart + confusion matrices
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG)
fig.suptitle("Step 2 – Baseline Comparison", fontsize=14, fontweight="bold", color=FG)

# Bar chart
ax = axes[0]
labels  = ["B0 Majority\nClass", "B1 Gender\nRule", "B2 LogReg\n(Pclass+Sex)"]
accs    = [b0_acc, b1_acc, b2_acc]
colors  = [PALETTE[3], PALETTE[0], PALETTE[1]]
bars = ax.bar(labels, accs, color=colors, edgecolor=GRID, width=0.5)
ax.axhline(b0_acc, color=PALETTE[3], linestyle="--", linewidth=1, alpha=0.5, label="Majority floor")
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy Comparison")
ax.legend(fontsize=8)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{acc*100:.1f}%", ha="center", fontsize=10)

# Confusion matrix B1
ax = axes[1]
cm = confusion_matrix(y_test, b1_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Died","Survived"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("B1: Gender Rule\nConfusion Matrix")
ax.set_facecolor(GRID)
for text in ax.texts:
    text.set_color(FG)

# Confusion matrix B2
ax = axes[2]
cm2 = confusion_matrix(y_test, b2_pred)
disp2 = ConfusionMatrixDisplay(cm2, display_labels=["Died","Survived"])
disp2.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("B2: LogReg (2 feat)\nConfusion Matrix")
ax.set_facecolor(GRID)
for text in ax.texts:
    text.set_color(FG)

plt.tight_layout()
out = os.path.join(PLOTS_DIR, "step2_baselines.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  ✔ Plot saved → {out}")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n  ┌─ Baseline Summary ─────────────────────────────────┐")
for name, acc in results.items():
    print(f"  │  {name:<25}  {acc*100:.2f}%")
print("  └────────────────────────────────────────────────────┘")
print(f"\n  Target: Beat B1 ({b1_acc*100:.2f}%) with proper feature engineering.")
print(f"  Target: Reach ≥80% test accuracy with a good model.\n")

# ── Append to experiment log ─────────────────────────────────────────────────
with open(LOG_PATH, "a") as f:
    f.write(f"""
---
## Step 2: Baselines

| Baseline | Description | Test Accuracy |
|----------|-------------|---------------|
| B0 | Majority class (everyone dies) | {b0_acc*100:.2f}% |
| B1 | Gender rule (female=survive, male=die) | {b1_acc*100:.2f}% |
| B2 | Logistic Regression (Pclass + Sex) | {b2_acc*100:.2f}% |

**Key takeaway:** A single feature (Sex) yields {b1_acc*100:.1f}% accuracy.
Feature engineering must push beyond this.

**Next target:** ≥80% with proper features + tuned model.
""")

print("  ✔ Experiment log updated.")
print("  STEP 2 COMPLETE — Baselines established.\n")
