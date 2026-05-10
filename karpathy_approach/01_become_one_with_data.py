"""
╔══════════════════════════════════════════════════════════════╗
║  STEP 1 — BECOME ONE WITH THE DATA                          ║
║  Karpathy's Rule #1: "Inspect your data before writing       ║
║  a single line of model code."                               ║
╚══════════════════════════════════════════════════════════════╝

WHY THIS STEP?
  Before touching a model, Karpathy insists you must deeply
  understand what you are working with. Most ML bugs stem from
  misunderstanding the data, not the model.

  Goals here:
    1. Know the shape and dtypes of every column.
    2. Find missing values and understand WHY they are missing.
    3. Spot obvious anomalies (duplicates, outliers, label leaks).
    4. Build intuition: which raw features seem most informative?

  This step produces NO model — only knowledge.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys

# ── paths ────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "titanic.csv")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── aesthetics ───────────────────────────────────────────────────────────────
BG, FG   = "#0f0f1a", "#cdd6f4"
GRID     = "#1e1e2e"
PALETTE  = ["#89b4fa", "#cba6f7", "#a6e3a1", "#f38ba8", "#fab387"]
sns.set_theme(style="dark")
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": GRID,
    "axes.edgecolor": "#6e6e8e", "axes.labelcolor": FG,
    "xtick.color": FG, "ytick.color": FG, "text.color": FG,
    "grid.color": "#313244",
})

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*65)
print("  STEP 1 │ BECOME ONE WITH THE DATA")
print("═"*65)

df = pd.read_csv(DATA_PATH)

# ── 1.1  Basic shape ─────────────────────────────────────────────────────────
print(f"\n  Rows: {df.shape[0]}   Columns: {df.shape[1]}")

# ── 1.2  Column types & missing values ───────────────────────────────────────
print("\n  ┌─ Columns, dtype, non-null count, % missing ─────────────────┐")
for col in df.columns:
    n_null = df[col].isnull().sum()
    pct    = n_null / len(df) * 100
    bar    = "█" * int(pct / 5)
    print(f"  │  {col:<15} {str(df[col].dtype):<10}  "
          f"null={n_null:>3} ({pct:5.1f}%)  {bar}")
print("  └─────────────────────────────────────────────────────────────┘")

# ── 1.3  Label distribution ──────────────────────────────────────────────────
vc = df["Survived"].value_counts()
print(f"\n  Target 'Survived':  0 (died)={vc[0]}  1 (survived)={vc[1]}")
print(f"  Class imbalance ratio: {vc[0]/vc[1]:.2f}:1  →  mild imbalance, no special handling needed yet.")

# ── 1.4  Sanity checks ───────────────────────────────────────────────────────
print(f"\n  Duplicate rows: {df.duplicated().sum()}")
print(f"  Age range      : {df['Age'].min():.1f} – {df['Age'].max():.1f} yrs")
print(f"  Fare range     : £{df['Fare'].min():.2f} – £{df['Fare'].max():.2f}")
print(f"  PassengerId unique: {df['PassengerId'].nunique()} / {len(df)}")

# ── 1.5  Sample rows for each class ──────────────────────────────────────────
print("\n  ► 3 random passengers who DIED:")
print(df[df["Survived"] == 0][["Name","Sex","Age","Pclass","Fare","Embarked"]].sample(3, random_state=1).to_string(index=False))
print("\n  ► 3 random passengers who SURVIVED:")
print(df[df["Survived"] == 1][["Name","Sex","Age","Pclass","Fare","Embarked"]].sample(3, random_state=1).to_string(index=False))

# ── 1.6  Visualise everything important in one dashboard ─────────────────────
fig = plt.figure(figsize=(18, 12), facecolor=BG)
fig.suptitle("Step 1 – Data Inspection Dashboard", fontsize=16, fontweight="bold", color=FG, y=0.98)

gs = fig.add_gridspec(3, 4, hspace=0.5, wspace=0.4)

# (a) Missing value bars
ax = fig.add_subplot(gs[0, 0])
missing = df.isnull().mean() * 100
missing = missing[missing > 0].sort_values(ascending=False)
ax.barh(missing.index, missing.values, color=PALETTE[3])
ax.set_title("% Missing per Column")
ax.set_xlabel("% missing")
for i, v in enumerate(missing.values):
    ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=8)

# (b) Survival count
ax = fig.add_subplot(gs[0, 1])
ax.bar(["Died", "Survived"], [vc[0], vc[1]], color=[PALETTE[3], PALETTE[2]])
ax.set_title("Survival Count")
for i, v in enumerate([vc[0], vc[1]]):
    ax.text(i, v + 5, str(v), ha="center")

# (c) Age histogram
ax = fig.add_subplot(gs[0, 2])
ax.hist(df["Age"].dropna(), bins=25, color=PALETTE[0], edgecolor=BG)
ax.axvline(df["Age"].median(), color=PALETTE[3], ls="--", label=f"median={df['Age'].median()}")
ax.set_title("Age Distribution")
ax.legend(fontsize=7)

# (d) Fare histogram (log)
ax = fig.add_subplot(gs[0, 3])
ax.hist(np.log1p(df["Fare"]), bins=30, color=PALETTE[1], edgecolor=BG)
ax.set_title("log(1+Fare) Distribution")
ax.set_xlabel("log(1+Fare)")

# (e) Pclass × Survival
ax = fig.add_subplot(gs[1, 0])
for cls, color in zip([1,2,3], PALETTE):
    sub = df[df["Pclass"]==cls]["Survived"]
    ax.bar(str(cls), sub.mean(), color=color, label=f"Class {cls}")
ax.set_title("Survival Rate by Pclass")
ax.set_ylabel("Rate")
ax.set_ylim(0, 1)

# (f) Sex × Survival
ax = fig.add_subplot(gs[1, 1])
sr = df.groupby("Sex")["Survived"].mean()
ax.bar(sr.index, sr.values, color=[PALETTE[3], PALETTE[0]])
ax.set_title("Survival Rate by Sex")
ax.set_ylabel("Rate")
ax.set_ylim(0, 1)
for i, (sex, rate) in enumerate(sr.items()):
    ax.text(i, rate + 0.02, f"{rate:.0%}", ha="center")

# (g) Embarked × Survival
ax = fig.add_subplot(gs[1, 2])
sr_e = df.groupby("Embarked")["Survived"].mean()
ax.bar(sr_e.index, sr_e.values, color=PALETTE[:3])
ax.set_title("Survival Rate by Embarked Port")
ax.set_ylabel("Rate")
ax.set_ylim(0, 1)

# (h) Age × Survival KDE
ax = fig.add_subplot(gs[1, 3])
for val, color, label in [(0, PALETTE[3], "Died"), (1, PALETTE[2], "Survived")]:
    data = df[df["Survived"]==val]["Age"].dropna()
    ax.hist(data, bins=20, alpha=0.5, color=color, label=label, density=True)
ax.set_title("Age by Survival")
ax.legend(fontsize=7)

# (i) Fare × Survival boxplot
ax = fig.add_subplot(gs[2, 0:2])
df_tmp = df.copy()
df_tmp["Survived_Label"] = df_tmp["Survived"].map({0:"Died",1:"Survived"})
sns.boxplot(data=df_tmp, x="Survived_Label", y="Fare",
            palette={"Died": PALETTE[3], "Survived": PALETTE[2]},
            ax=ax, linewidth=1, flierprops=dict(marker=".", markersize=3))
ax.set_title("Fare by Survival")

# (j) Correlation heatmap
ax = fig.add_subplot(gs[2, 2:4])
numeric = df[["Survived","Pclass","Age","SibSp","Parch","Fare"]].corr()
sns.heatmap(numeric, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=0.5, linecolor=BG, ax=ax, cbar_kws={"shrink":0.7})
ax.set_title("Feature Correlation Matrix")

out = os.path.join(PLOTS_DIR, "step1_data_dashboard.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()

print(f"\n  ✔ Dashboard saved → {out}")

# ── 1.7  Write key observations to a log ─────────────────────────────────────
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_log.md")
with open(log_path, "w") as f:
    f.write(f"""# Titanic – Karpathy Auto-Research Experiment Log

---
## Step 1: Become One With The Data

**Dataset:** 891 rows × 12 columns

### Missing Values
| Column   | Missing | %      | Strategy |
|----------|---------|--------|----------|
| Cabin    | 687     | 77.1%  | Drop or extract deck letter |
| Age      | 177     | 19.9%  | Impute with median (or model-based) |
| Embarked | 2       | 0.2%   | Fill with mode ('S') |

### Key Observations
- Survival rate: {df['Survived'].mean()*100:.1f}%  →  class imbalance is mild
- **Sex** is the strongest raw signal: female survival {df[df['Sex']=='female']['Survived'].mean()*100:.0f}% vs male {df[df['Sex']=='male']['Survived'].mean()*100:.0f}%
- **Pclass** is strongly inversely correlated with survival (-0.34)
- **Fare** positively correlated (+0.26), likely a proxy for class/wealth
- Children under 10 survived at {df[df['Age']<10]['Survived'].mean()*100:.0f}%
- No duplicates; PassengerId is unique → safe to drop
- Cabin has 77% missing → drop for baseline; could extract deck letter later

### Features to Build
- `Title` extracted from Name (Mr, Mrs, Miss, Master, Rare)
- `FamilySize` = SibSp + Parch + 1
- `IsAlone` = 1 if FamilySize == 1
- `AgeBand` = binned age (child/teen/adult/senior)
- `FareBand` = log-transformed or binned fare

### Next Step
→ Establish a dumb baseline (majority class predictor)
""")

print(f"  ✔ Experiment log started → {log_path}")
print("\n  STEP 1 COMPLETE — You now understand your data.\n")
