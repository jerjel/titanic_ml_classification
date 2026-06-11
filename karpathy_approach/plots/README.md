# 🚢 Titanic AutoResearch — A Learning Guide

> *"Give an AI agent a problem and let it experiment overnight. You wake up to a log of experiments and a better model."*
> — Inspired by Andrej Karpathy's AutoResearch

---

## 📖 Table of Contents

1. [What Is AutoResearch?](#1-what-is-autoresearch)
2. [The Titanic Problem — Our Test Case](#2-the-titanic-problem--our-test-case)
3. [Machine Learning Concepts You Need to Know](#3-machine-learning-concepts-you-need-to-know)
4. [Project Structure](#4-project-structure)
5. [How the Code Works — File by File](#5-how-the-code-works--file-by-file)
6. [Running Your First Experiment](#6-running-your-first-experiment)
7. [The Experiment Loop — How AutoResearch Works](#7-the-experiment-loop--how-autoresearch-works)
8. [Guided Experiments — Try These Yourself](#8-guided-experiments--try-these-yourself)
9. [Understanding Your Results](#9-understanding-your-results)
10. [Running AutoResearch with an AI Agent](#10-running-autoresearch-with-an-ai-agent)
11. [Glossary](#11-glossary)

---

## 1. What Is AutoResearch?

### The Old Way of Doing ML Research

Imagine you are a scientist trying to find the best recipe for a cake. You would:
1. Try Recipe A → taste it → write down the score
2. Change one ingredient → try Recipe B → taste it → write down the score
3. Keep whichever recipe tasted better
4. Repeat, over and over, for hours

Machine learning (ML) research works the same way. A researcher tries different models and settings, measures how good each one is, keeps improvements, and throws away failures. This is slow and boring.

### The AutoResearch Way

**Andrej Karpathy** (former AI Director at Tesla, co-founder of OpenAI) had a simpler idea in 2026:

> What if an **AI agent** did the experimenting for you, automatically, while you sleep?

You set up three files:

| File | Who writes it | What it does |
|---|---|---|
| `prepare.py` | You (once, at the start) | Loads data, defines the scoring rule. Never changes. |
| `train.py` | The AI agent | The model to improve. Agent edits this every experiment. |
| `program.md` | You (research directions) | Instructions telling the agent what ideas to try. |

The agent runs this loop **all night**:

```
READ results.jsonl → THINK of one idea → EDIT train.py
→ RUN python train.py → CHECK score
→ If better: keep it (git commit) ✅
→ If worse: revert (git checkout) ❌
→ REPEAT
```

By morning, you have a log of 20–100 experiments and (hopefully) a much better model.

---

## 2. The Titanic Problem — Our Test Case

### What Happened?

On April 15, 1912, the RMS Titanic sank in the Atlantic Ocean. Of the 891 passengers in our dataset, **only 342 survived** (about 38%).

### The Question

Can a computer program look at a passenger's information — their age, gender, ticket class, etc. — and **predict whether they survived or died**?

This is a **classification problem**: for each passenger, the answer is either 0 (died) or 1 (survived).

### The Data

Each row in `titanic.csv` represents one passenger:

| Column | What it means | Example |
|---|---|---|
| `Survived` | Did they survive? (our **answer** to predict) | 0 or 1 |
| `Pclass` | Ticket class (1=First, 2=Second, 3=Third) | 3 |
| `Sex` | Gender | male |
| `Age` | Age in years | 22 |
| `SibSp` | Number of siblings/spouses aboard | 1 |
| `Parch` | Number of parents/children aboard | 0 |
| `Fare` | Ticket price paid | 7.25 |
| `Embarked` | Port where they boarded (C, Q, or S) | S |

**Key insight from history:** Women and children were given priority in lifeboats. First-class passengers had better access. This is real signal the model can learn from.

---

## 3. Machine Learning Concepts You Need to Know

### 3.1 Features vs. Labels

Think of it like a school report card:

- **Features (X)** = the input information the model uses to make a prediction.
- **Label (y)** = the answer we want to predict.

In our project:
```python
feature_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = df[feature_cols].values   # a table of numbers
y = df["Survived"].values     # a list of 0s and 1s
```

### 3.2 Training vs. Validation Split

We never let the model see the answers for data it will be tested on. It's like a teacher giving practice problems (training) and then a real test (validation) with different questions.

```python
# 80% of passengers → training set (model learns from these)
# 20% of passengers → validation set (we test the model here)
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,        # 20% for validation
    random_state=42,      # fixed seed → same split every time = fair comparison
    stratify=y            # keep same survival ratio in both sets
)
```

> ⚠️ **Why a fixed seed matters:** If we used a different random split every time, a model that got "lucky" with an easier split would look better. The fixed seed (42) means every experiment is tested on the **exact same 179 passengers** — making results truly comparable.

### 3.3 What Is a Model?

A model is a mathematical function that takes features (X) as input and outputs a prediction.

- **Logistic Regression** — Draws a straight line to separate survivors from non-survivors.
- **Random Forest** — Builds many decision trees and takes a vote. Like 100 different judges each deciding independently — the majority wins.
- **Gradient Boosting** — Builds trees one at a time, each learning from the mistakes of the previous one.

### 3.4 What Is Accuracy?

```
Accuracy = (Number of correct predictions) / (Total predictions)
```

If the model correctly predicted 143 out of 179 validation passengers:
```
Accuracy = 143 / 179 = 0.7989 = 79.89%
```

**Our baseline score: 0.7989. Our goal is to beat this.**

### 3.5 What Is AUC?

AUC stands for **Area Under the Curve**. While accuracy counts right/wrong answers, AUC measures how *confident* the model's predictions are.

- **AUC = 0.5** → no better than random guessing (coin flip)
- **AUC = 1.0** → perfect predictions
- **AUC = 0.85** → good model (our baseline: 0.8519)

### 3.6 What Is a Pipeline?

A Pipeline chains steps together so data flows through them automatically:

```python
model = Pipeline([
    ("scaler", StandardScaler()),        # Step 1: normalize the numbers
    ("clf",    LogisticRegression()),    # Step 2: train the classifier
])
```

Without scaling, `Fare` (0–512) would dominate `Pclass` (1–3), even if Pclass matters more. `StandardScaler` converts everything to the same scale.

### 3.7 What Is Feature Engineering?

Creating new, smarter columns from the raw data.

**Example:** `FamilySize = SibSp + Parch + 1`. A family of 5 had a harder time getting a lifeboat than a solo traveler. This captures that logic in a single number.

### 3.8 What Is Hyperparameter Tuning?

A model has settings called **hyperparameters** you choose before training:
- Number of trees in a Random Forest (`n_estimators`)
- Maximum depth of each tree (`max_depth`)

**Optuna** automates the search:

```python
import optuna

def objective(trial):
    n = trial.suggest_int("n_estimators", 100, 500)
    d = trial.suggest_int("max_depth", 3, 10)
    model = RandomForestClassifier(n_estimators=n, max_depth=d)
    return cross_val_score(model, X_train, y_train, cv=5).mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, timeout=60)   # search for 60 seconds
print(study.best_params)
```

---

## 4. Project Structure

```
autoresearch/
│
├── prepare.py      ← 🔒 FIXED — Data loading, evaluation rules. NEVER modify.
├── train.py        ← ✏️  AGENT modifies this every experiment.
├── program.md      ← 📋 YOU write research directions here.
├── results.jsonl   ← 📊 Auto-generated log of every experiment.
└── README.md       ← 📖 This file.
```

**The Golden Rule:** Only `train.py` ever changes. This ensures every experiment is a fair comparison — same data, same evaluation, same scoring rule.

---

## 5. How the Code Works — File by File

### 5.1 `prepare.py` — The Foundation (Never Touch This)

**① Loads and cleans the data:**
```python
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Age"] = df["Age"].fillna(df["Age"].median())  # fill missing ages with the median
    df["Embarked"] = df["Embarked"].fillna("S")        # fill 2 missing ports
    df.drop(columns=["Cabin", "Name", "Ticket"], ...)  # drop unhelpful columns
```

**② Encodes text as numbers** (models only understand numbers):
```python
le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])       # "male" → 1, "female" → 0
df["Embarked"] = le.fit_transform(df["Embarked"])  # "C"→0, "Q"→1, "S"→2
```

**③ Always uses the same train/val split:**
```python
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# 712 passengers for training, 179 for validation — same every time
```

**④ The canonical `evaluate()` function:**
```python
def evaluate(model, X_val, y_val) -> dict:
    y_pred = model.predict(X_val)
    acc = float(accuracy_score(y_val, y_pred))
    auc = float(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
    return {"val_accuracy": acc, "val_auc": auc, ...}
```

**⑤ Saves results and git-commits the code:**
```python
def log_result(experiment_name, metrics, notes):
    # Writes one line to results.jsonl for every experiment

def git_save(message):
    # Runs: git add train.py && git commit -m "..."
    # Each improvement is permanently saved in git history
```

### 5.2 `train.py` — The Experiment File

Every experiment follows this exact structure:

```python
# 1. IDENTITY — unique name for this experiment
experiment_name = "baseline_logistic_regression"
notes = "Describe what you changed and why."

# 2. DATA — always loaded the same way
X_train, X_val, y_train, y_val = load_data()

# 3. MODEL — THIS IS WHAT YOU CHANGE!
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=1000, random_state=42)),
])

# 4. TRAIN
model.fit(X_train, y_train)

# 5. EVALUATE — Fixed, never changes
metrics = evaluate(model, X_val, y_val)

# 6. LOG & SAVE — Fixed, never changes
log_result(experiment_name, metrics, notes)
git_save(f"[autoresearch] {experiment_name} acc={metrics['val_accuracy']:.4f}")
```

### 5.3 `results.jsonl` — The Experiment Log

Every run appends one line:

```json
{"experiment": "baseline_logistic_regression",  "val_accuracy": 0.7989, "val_auc": 0.8519}
{"experiment": "random_forest_v1",               "val_accuracy": 0.8268, "val_auc": 0.8821}
{"experiment": "rf_with_family_size",            "val_accuracy": 0.8436, "val_auc": 0.9012}
```

This is your **research journal** — a permanent record of every idea tried.

---

## 6. Running Your First Experiment

```bash
# Step 1: Verify setup
cd autoresearch/
python prepare.py
# Expected: "✓ Data pipeline OK"

# Step 2: Run the baseline
python train.py
# Expected:
# =======================================================
#   Experiment : baseline_logistic_regression
#   Val Accuracy : 0.7989
#   Val AUC      : 0.8519
# =======================================================

# Step 3: Check the log
cat results.jsonl
```

**Goal: Beat 0.7989 accuracy.**

---

## 7. The Experiment Loop — How AutoResearch Works

```
┌─────────────────────────────────────────────────┐
│  1. Read results.jsonl → What's the best so far?│
│  2. Think of ONE improvement idea               │
│  3. Edit train.py with the idea                 │
│  4. Run: python train.py                        │
│  5. Did val_accuracy improve?                   │
│       YES → git automatically commits it ✅     │
│       NO  → git checkout train.py (revert) ❌   │
│  6. Repeat from step 1                          │
└─────────────────────────────────────────────────┘
```

**Why git?** It gives you a complete safety net. You can always return to any previous experiment.

```bash
# See all experiments in order:
git log --oneline

# Go back to a specific experiment:
git checkout <commit-hash> -- train.py
```

---

## 8. Guided Experiments — Try These Yourself

### Experiment 1: Switch to Random Forest

**Why:** Logistic Regression draws straight lines. Titanic survival had complex interactions (e.g., a woman in 3rd class vs. 1st class). Random Forests capture non-linear patterns.

```python
experiment_name = "random_forest_baseline"
notes = "RandomForest 200 trees, no feature engineering yet."

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=200,    # 200 decision trees
        max_depth=7,         # each tree at most 7 levels deep
        min_samples_leaf=3,  # each leaf needs at least 3 passengers
        random_state=42
    )),
])
```

Run it: `python train.py`
If better → keep! If not → `git checkout train.py`

---

### Experiment 2: Add FamilySize Feature

**Why:** Solo travelers may have found lifeboats more easily; large families struggled.

```python
experiment_name = "rf_with_family_size"
notes = "Added FamilySize and IsAlone features."

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

df = load_raw_df()

# New features
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)

le = LabelEncoder()
df["Sex"]      = le.fit_transform(df["Sex"])
df["Embarked"] = le.fit_transform(df["Embarked"])

feature_cols = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","FamilySize","IsAlone"]
X = df[feature_cols].values.astype("float32")
y = df["Survived"].values.astype("int32")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=200, max_depth=7, min_samples_leaf=3, random_state=42)),
])
```

---

### Experiment 3: Extract Title from Name

**Why:** "Master" (boys under ~12) had high survival. "Rev" (priests) had very low survival. These are hidden inside the Name column.

```python
experiment_name = "rf_title_feature"
notes = "Extracted Title from Name. Grouped rare titles."

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

df = load_raw_df()
raw = pd.read_csv("../titanic.csv")
df["Name"] = raw["Name"]

# Extract title
df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)

# Group rare titles
rare = ["Lady","Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"]
df["Title"] = df["Title"].replace(rare, "Rare")
df["Title"] = df["Title"].replace({"Mlle":"Miss","Ms":"Miss","Mme":"Mrs"})

title_map = {"Mr":1, "Miss":2, "Mrs":3, "Master":4, "Rare":5}
df["Title"] = df["Title"].map(title_map).fillna(0).astype(int)

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)

le = LabelEncoder()
df["Sex"]      = le.fit_transform(df["Sex"])
df["Embarked"] = le.fit_transform(df["Embarked"])

feature_cols = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","FamilySize","IsAlone","Title"]
X = df[feature_cols].values.astype("float32")
y = df["Survived"].values.astype("int32")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=300, max_depth=7, min_samples_leaf=3, random_state=42)),
])
```

---

### Experiment 4: Hyperparameter Tuning with Optuna

**Why:** The difference between `max_depth=5` and `max_depth=9` can mean 1–2% accuracy.

```python
experiment_name = "rf_optuna_tuned"
notes = "Optuna 60-second search for best RandomForest hyperparameters."

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

X_train, X_val, y_train, y_val = load_data()

def objective(trial):
    params = {
        "n_estimators"    : trial.suggest_int("n_estimators", 100, 600),
        "max_depth"       : trial.suggest_int("max_depth", 3, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features"    : trial.suggest_categorical("max_features", ["sqrt","log2"]),
    }
    clf = RandomForestClassifier(**params, random_state=42)
    return cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy").mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, timeout=60)

best = study.best_params
print(f"Best params: {best}")
print(f"Best CV accuracy: {study.best_value:.4f}")

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(**best, random_state=42)),
])
```

---

### Experiment 5: Soft-Voting Ensemble

**Why:** Different models make different mistakes. Combining them takes the best of each.

```python
experiment_name = "soft_voting_ensemble"
notes = "Soft-voting: LogReg + RandomForest + GradientBoosting."

X_train, X_val, y_train, y_val = load_data()

from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier

lr = Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=1000, random_state=42))])
rf = RandomForestClassifier(n_estimators=300, max_depth=7, min_samples_leaf=3, random_state=42)
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)

model = VotingClassifier(
    estimators=[("lr", lr), ("rf", rf), ("gb", gb)],
    voting="soft"   # use predicted probabilities, not just votes
)
```

---

## 9. Understanding Your Results

### Print a Leaderboard

```python
import json, pandas as pd

results = [json.loads(line) for line in open("results.jsonl")]
df = pd.DataFrame(results).sort_values("val_accuracy", ascending=False)
print(df[["experiment","val_accuracy","val_auc"]].to_string(index=False))
```

Expected output after several experiments:
```
               experiment  val_accuracy  val_auc
     soft_voting_ensemble        0.8547   0.9081
         rf_optuna_tuned         0.8492   0.9034
          rf_title_feature       0.8436   0.9012
      rf_with_family_size        0.8324   0.8903
      random_forest_baseline     0.8268   0.8821
baseline_logistic_regression     0.7989   0.8519
```

---

## 10. Running AutoResearch with an AI Agent

Open this project in an AI coding assistant and say:

```
Read program.md and results.jsonl.
Run the experiment loop to maximise val_accuracy on the Titanic dataset.
Try one idea at a time. Run train.py, check the score, keep improvements,
revert failures with git checkout train.py. Stop after 10 experiments
or when accuracy >= 0.85.
```

The agent will do everything automatically. You just come back and read `results.jsonl`.

**What you control (as the human researcher)** — edit `program.md` to:
- Add new ideas to try
- Change the target metric
- Set constraints or time limits

This is the core insight: **you write the research strategy, the agent executes the experiments.**

---

## 11. Glossary

| Term | Plain English Meaning |
|---|---|
| **Classification** | Predicting which category something belongs to (survived / died) |
| **Feature** | A column of input data used to make predictions |
| **Label / Target** | The answer we want to predict (0 or 1) |
| **Training set** | Data the model learns from (80%) |
| **Validation set** | Hidden data used to check how good the model is (20%) |
| **Overfitting** | Model memorises training data but fails on new data |
| **Accuracy** | % of predictions that were correct |
| **AUC** | How well the model ranks survivors above non-survivors (0.5–1.0) |
| **Hyperparameter** | A setting chosen before training (e.g., number of trees) |
| **Feature Engineering** | Creating new, smarter columns from existing data |
| **Pipeline** | A chain of processing steps applied in order |
| **StandardScaler** | Converts features to the same numeric scale |
| **Random Forest** | Many decision trees that vote together |
| **Gradient Boosting** | Trees built one at a time, each fixing the last one's mistakes |
| **Ensemble** | Combining multiple models for better accuracy |
| **Optuna** | A library that automatically searches for the best hyperparameters |
| **git commit** | Saves a code snapshot you can return to |
| **git checkout** | Reverts a file to a previous saved state |
| **JSONL** | A file where each line is a separate JSON record |
| **AutoResearch** | An AI agent that runs the experiment loop automatically |

---

## 🎓 Summary

```
AutoResearch = Fixed Rules + Iterable Code + AI Agent + Git Safety Net

prepare.py  → Fixed rules: same data split, same evaluation every time
train.py    → Only file that changes: model, features, hyperparameters
program.md  → Your research strategy written in plain English
results.jsonl → Permanent journal of every experiment ever tried
git         → Safety net: keep improvements, revert failures
```

**Start here → beat 79.89% → target 85%+ → the agent does the work.**

---

*Built with inspiration from Karpathy's [autoresearch](https://github.com/karpathy/autoresearch). Adapted for classical ML + Titanic classification.*
