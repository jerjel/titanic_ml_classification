"""
train.py — The single file the AI agent modifies.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPERIMENT: Baseline — Logistic Regression with default features

The agent is free to change:
  - The model class and its hyperparameters
  - Feature engineering (use load_raw_df() to access the full DataFrame)
  - Preprocessing (scaling, encoding, imputation)
  - Ensembling
  - Anything else that might improve val_accuracy

DO NOT change:
  - The import of prepare.py helpers
  - The final call to evaluate(), log_result(), and git_save()
  - The experiment_name variable (update it for each new experiment)
"""

import sys
from pathlib import Path

# Add parent dir so prepare.py is importable
sys.path.insert(0, str(Path(__file__).parent))
from prepare import load_data, load_raw_df, evaluate, log_result, git_save

# ── Imports the agent may need ────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import (RandomForestClassifier,
                                     GradientBoostingClassifier,
                                     VotingClassifier)
from sklearn.svm             import SVC
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import cross_val_score

# ── Experiment identity ────────────────────────────────────────────────────────
experiment_name = "baseline_logistic_regression"
notes = "Default LogisticRegression on 7 base features, no extra engineering."

# ── Data ──────────────────────────────────────────────────────────────────────
X_train, X_val, y_train, y_val = load_data()

# ── Model definition ──────────────────────────────────────────────────────────
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=1000, random_state=42)),
])

# ── Training ──────────────────────────────────────────────────────────────────
model.fit(X_train, y_train)

# ── Evaluation ────────────────────────────────────────────────────────────────
metrics = evaluate(model, X_val, y_val)

print(f"\n{'='*55}")
print(f"  Experiment : {experiment_name}")
print(f"  Val Accuracy : {metrics['val_accuracy']:.4f}")
print(f"  Val AUC      : {metrics['val_auc']:.4f}")
print(f"{'='*55}\n")

# ── Log & save ────────────────────────────────────────────────────────────────
entry = log_result(experiment_name, metrics, notes)
git_save(f"[autoresearch] {experiment_name} acc={metrics['val_accuracy']:.4f}")

print("Result logged to results.jsonl")
