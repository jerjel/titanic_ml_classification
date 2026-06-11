"""
prepare.py — Fixed constants, data loading, preprocessing, and evaluation.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DO NOT MODIFY — This file is the fixed foundation that train.py builds on.
The AI agent only modifies train.py. This file handles:
  - Data loading from the raw CSV
  - Train/val split (fixed seed for fair comparison)
  - Feature preprocessing into numpy arrays
  - The canonical evaluate() function that returns a single comparable metric
"""

import os
import json
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_PATH    = Path(__file__).parent.parent / "titanic.csv"
RESULTS_FILE = Path(__file__).parent / "results.jsonl"
RANDOM_SEED  = 42
VAL_SIZE     = 0.2          # 20% held-out validation set

# ── Data Loading & Preprocessing ──────────────────────────────────────────────

def load_data():
    """
    Loads and preprocesses the Titanic dataset.
    Returns X_train, X_val, y_train, y_val as numpy arrays.
    The preprocessing here is intentionally minimal — the agent can do
    additional feature engineering inside train.py using the raw DataFrame.
    """
    df = pd.read_csv(DATA_PATH)

    # ── Basic cleaning (always applied) ──────────────────────────────────────
    df["Age"]      = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna("S")
    df["Fare"]     = df["Fare"].fillna(df["Fare"].median())

    # Drop columns with too many unique values or irrelevant info
    df.drop(columns=["Cabin", "Name", "Ticket"], errors="ignore", inplace=True)

    # Encode categoricals
    le = LabelEncoder()
    df["Sex"]      = le.fit_transform(df["Sex"])          # male=1, female=0
    df["Embarked"] = le.fit_transform(df["Embarked"])     # C=0, Q=1, S=2

    # ── Feature / label split ─────────────────────────────────────────────────
    feature_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["Survived"].values.astype(np.int32)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    return X_train, X_val, y_train, y_val


def load_raw_df():
    """Returns the raw DataFrame for use by the agent in feature engineering."""
    df = pd.read_csv(DATA_PATH)
    df["Age"]      = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna("S")
    df["Fare"]     = df["Fare"].fillna(df["Fare"].median())
    df.drop(columns=["Cabin", "Name", "Ticket"], errors="ignore", inplace=True)
    return df


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(model, X_val, y_val) -> dict:
    """
    Evaluates a fitted sklearn-compatible classifier.
    Returns a dict with the canonical metric (val_accuracy) plus extras.
    HIGHER val_accuracy is better. This is the single number the agent optimises.
    """
    from sklearn.metrics import (
        accuracy_score, roc_auc_score,
        classification_report, confusion_matrix
    )
    y_pred = model.predict(X_val)
    try:
        y_prob = model.predict_proba(X_val)[:, 1]
        auc    = float(roc_auc_score(y_val, y_prob))
    except Exception:
        auc    = float("nan")

    acc = float(accuracy_score(y_val, y_pred))
    return {
        "val_accuracy" : acc,
        "val_auc"      : auc,
        "report"       : classification_report(y_val, y_pred, output_dict=True),
        "confusion"    : confusion_matrix(y_val, y_pred).tolist(),
    }


# ── Result Logging ─────────────────────────────────────────────────────────────

def log_result(experiment_name: str, metrics: dict, notes: str = ""):
    """Appends a result entry to results.jsonl so all runs are traceable."""
    entry = {
        "experiment"  : experiment_name,
        "val_accuracy": metrics["val_accuracy"],
        "val_auc"     : metrics["val_auc"],
        "notes"       : notes,
    }
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


# ── Git snapshot ───────────────────────────────────────────────────────────────

def git_save(message: str):
    """Saves the current state of train.py as a git commit."""
    try:
        subprocess.run(["git", "add", "train.py"], cwd=Path(__file__).parent, check=True)
        subprocess.run(["git", "commit", "-m", message], cwd=Path(__file__).parent, check=True)
    except subprocess.CalledProcessError:
        pass   # Not fatal — just won't have a clean commit history


if __name__ == "__main__":
    print("prepare.py — verifying data pipeline …")
    X_train, X_val, y_train, y_val = load_data()
    print(f"  Train size : {X_train.shape[0]}")
    print(f"  Val size   : {X_val.shape[0]}")
    print(f"  Features   : {X_train.shape[1]}")
    print(f"  Survival rate (train): {y_train.mean():.2%}")
    print("  ✓ Data pipeline OK")
