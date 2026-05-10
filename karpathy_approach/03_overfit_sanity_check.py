"""
╔══════════════════════════════════════════════════════════════╗
║  STEP 3 — OVERFIT A TINY BATCH                              ║
║  Karpathy's Rule #3: "Before scaling up, make sure your     ║
║  pipeline can overfit a tiny sample. If it can't, your      ║
║  pipeline is broken."                                        ║
╚══════════════════════════════════════════════════════════════╝

WHY THIS STEP?
  This is a debugging / sanity-check step unique to Karpathy's
  recipe. The idea is:
    • Take a tiny subset (e.g., 20 samples).
    • Train until the model memorises them (near 100% train acc).
    • If it cannot, the model, loss, or data pipeline is wrong.

  For classical ML this translates to:
    • Remove all regularisation.
    • Confirm train accuracy → ~100% on small batch.
    • This validates: data loading, feature encoding,
      train/predict pipeline, and metric calculation.

  WHY IT MATTERS:
    A model that cannot overfit 20 examples has no capacity to
    learn anything — you would be wasting time tuning it.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "titanic.csv")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
LOG_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_log.md")
os.makedirs(PLOTS_DIR, exist_ok=True)

BG, FG = "#0f0f1a", "#cdd6f4"
GRID   = "#1e1e2e"
PAL    = ["#89b4fa", "#a6e3a1", "#f38ba8", "#fab387", "#cba6f7"]
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": GRID,
    "axes.edgecolor": "#6e6e8e", "axes.labelcolor": FG,
    "xtick.color": FG, "ytick.color": FG, "text.color": FG,
    "grid.color": "#313244",
})

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*65)
print("  STEP 3 │ OVERFIT A TINY BATCH (Pipeline Sanity Check)")
print("═"*65)

# ── Minimal feature engineering (needed for ALL later steps) ─────────────────
def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproducible feature engineering function used by all steps.
    Returns a cleaned, encoded feature DataFrame.
    """
    df = raw_df.copy()

    # Drop low-signal / ID columns
    df.drop(["PassengerId", "Ticket"], axis=1, inplace=True)

    # ── Title extraction (from Name) ─────────────────────────────────────────
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    TITLE_MAP = {
        "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
        "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
        "Mlle": "Miss", "Mme": "Mrs", "Ms": "Miss", "Don": "Rare",
        "Lady": "Rare", "Countess": "Rare", "Jonkheer": "Rare",
        "Sir": "Rare", "Capt": "Rare"
    }
    df["Title"] = df["Title"].map(TITLE_MAP).fillna("Rare")
    df.drop("Name", axis=1, inplace=True)

    # ── Age imputation (per title group median) ───────────────────────────────
    for title in df["Title"].unique():
        mask = (df["Title"] == title) & df["Age"].isnull()
        fill = df[df["Title"] == title]["Age"].median()
        df.loc[mask, "Age"] = fill
    df["Age"].fillna(df["Age"].median(), inplace=True)

    # ── Cabin: extract deck letter; NaN → 'Unknown' ───────────────────────────
    df["Deck"] = df["Cabin"].str[0].fillna("U")
    df.drop("Cabin", axis=1, inplace=True)

    # ── Family features ───────────────────────────────────────────────────────
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)

    # ── Fare: log transform ───────────────────────────────────────────────────
    df["Fare_log"] = np.log1p(df["Fare"])

    # ── Age bands ─────────────────────────────────────────────────────────────
    df["AgeBand"] = pd.cut(df["Age"], bins=[0, 12, 18, 35, 60, 100],
                           labels=[0, 1, 2, 3, 4]).astype(int)

    # ── Embarked fill ─────────────────────────────────────────────────────────
    df["Embarked"].fillna("S", inplace=True)

    # ── Encode categoricals ───────────────────────────────────────────────────
    for col in ["Sex", "Embarked", "Title", "Deck"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # ── Final feature set ─────────────────────────────────────────────────────
    FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare_log",
                "Embarked", "Title", "Deck", "FamilySize", "IsAlone", "AgeBand"]
    return df[FEATURES + ["Survived"]]

# ─────────────────────────────────────────────────────────────────────────────
raw = pd.read_csv(DATA_PATH)
data = build_features(raw)
X = data.drop("Survived", axis=1)
y = data["Survived"]

print(f"\n  Features after engineering: {list(X.columns)}\n")

# ── 3.1  Overfit 20 samples with an un-regularised decision tree ──────────────
TINY_N = 20
print(f"  [3.1] Overfitting {TINY_N} samples with DecisionTree (max_depth=None)")
X_tiny = X.iloc[:TINY_N]
y_tiny = y.iloc[:TINY_N]

dt_overfit = DecisionTreeClassifier(max_depth=None, random_state=42)   # no regularisation
dt_overfit.fit(X_tiny, y_tiny)
train_acc = accuracy_score(y_tiny, dt_overfit.predict(X_tiny))
print(f"      Train accuracy on {TINY_N} samples: {train_acc*100:.2f}%")
assert train_acc == 1.0, "Pipeline broken: model cannot memorise 20 samples!"
print("      ✔ PASS — model memorised tiny batch. Pipeline is correct.")

# ── 3.2  Learning curve: train vs test accuracy vs training set size ──────────
print(f"\n  [3.2] Learning curve (RandomForest, vary training size)")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

sizes  = [10, 20, 30, 50, 75, 100, 150, 200, 300, 400, len(X_train)]
tr_acc = []
te_acc = []

for n in sizes:
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train.iloc[:n], y_train.iloc[:n])
    tr_acc.append(accuracy_score(y_train.iloc[:n], rf.predict(X_train.iloc[:n])))
    te_acc.append(accuracy_score(y_test, rf.predict(X_test)))

# ── 3.3  Overfit detector: full train vs test gap ─────────────────────────────
rf_full = RandomForestClassifier(n_estimators=100, random_state=42)
rf_full.fit(X_train, y_train)
full_train_acc = accuracy_score(y_train, rf_full.predict(X_train))
full_test_acc  = accuracy_score(y_test,  rf_full.predict(X_test))
gap = full_train_acc - full_test_acc
print(f"      Full training set  train acc: {full_train_acc*100:.2f}%")
print(f"      Full training set  test acc : {full_test_acc*100:.2f}%")
print(f"      Generalisation gap          : {gap*100:.2f}%")
if gap > 0.10:
    print("      ⚠ Large gap detected → model is overfitting. Add regularisation.")
else:
    print("      ✔ Gap is acceptable — model generalises well.")

# ── Plot learning curve ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
ax.plot(sizes, [a*100 for a in tr_acc], "o-", color=PAL[1], label="Train accuracy")
ax.plot(sizes, [a*100 for a in te_acc], "s-", color=PAL[2], label="Test accuracy")
ax.fill_between(sizes,
                [a*100 for a in tr_acc],
                [a*100 for a in te_acc],
                alpha=0.15, color=PAL[2], label="Generalisation gap")
ax.axhline(full_test_acc*100, color=PAL[3], ls="--", linewidth=1.2,
           label=f"Full test acc: {full_test_acc*100:.1f}%")
ax.set_title("Step 3 – Learning Curve (Overfitting Check)")
ax.set_xlabel("Training set size")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(60, 102)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

out = os.path.join(PLOTS_DIR, "step3_learning_curve.png")
plt.tight_layout()
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  ✔ Learning curve saved → {out}")

# ── Append to experiment log ─────────────────────────────────────────────────
with open(LOG_PATH, "a") as f:
    f.write(f"""
---
## Step 3: Overfit Sanity Check

### Feature Engineering Applied
- `Title` extracted from Name → 5 categories (Mr, Mrs, Miss, Master, Rare)
- `FamilySize` = SibSp + Parch + 1
- `IsAlone` flag
- `AgeBand` (0–4 bins)
- `Fare_log` = log(1+Fare)
- `Deck` from first Cabin letter (U = unknown)
- Age imputed per Title-group median

### Overfit Check Result
- 20-sample DecisionTree train accuracy: {train_acc*100:.0f}% ✔
- Full RF train accuracy: {full_train_acc*100:.2f}%
- Full RF test accuracy:  {full_test_acc*100:.2f}%
- Gap: {gap*100:.2f}% → {'acceptable' if gap <= 0.10 else 'overfitting detected'}

### Next Step
→ Hyperparameter tuning + cross-validation to close the gap
""")

print("  ✔ Experiment log updated.")
print("  STEP 3 COMPLETE — Pipeline validated. Features confirmed.\n")
