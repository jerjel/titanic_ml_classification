# program.md — Research instructions for the AI agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# This is the "human-written" file. It defines the research direction.
# The agent reads this file to understand what to do, then modifies train.py
# and runs it to evaluate each idea.

## Project

You are an autonomous ML research agent working on the **Titanic survival
classification** problem. Your mission is to maximise **val_accuracy** on the
held-out validation set (20% of the data, fixed seed, stratified split).

## Workflow (follow exactly)

1. Read `results.jsonl` to see what has already been tried.
2. Think of ONE concrete improvement to `train.py` (see ideas below).
3. Edit only `train.py`. Do **NOT** modify `prepare.py`.
4. Run: `python train.py`
5. Check the printed `val_accuracy`. 
   - If it is **higher** than the current best → keep the change, it is
     automatically git-committed.
   - If it is **equal or lower** → revert: `git checkout train.py`
6. Repeat from step 1.

## The metric

`val_accuracy` — the fraction of passengers correctly classified as
survived/died on the fixed validation set. Higher is better.

Current best to beat: **0.7933** (baseline Logistic Regression).

## Ideas to explore (roughly ordered by expected impact)

### Feature Engineering
- `Title` — extract title from Name (Mr, Mrs, Miss, Master, Rev, Dr, …).
  Map rare titles to "Rare". Encode ordinally.
- `FamilySize` = SibSp + Parch + 1
- `IsAlone` = 1 if FamilySize == 1
- `FareBin` — bin Fare into quartiles (pd.qcut)
- `AgeBin` — bin Age into bands (child < 12, teen, adult, senior)
- `Deck` — first letter of Cabin when available (mostly NaN, handle carefully)
- `TicketFreq` — how many passengers share the same ticket number

### Model improvements
- `RandomForestClassifier(n_estimators=300, max_depth=7, min_samples_leaf=3)`
- `GradientBoostingClassifier(n_estimators=200, learning_rate=0.05)`
- `XGBClassifier` (if installed: `pip install xgboost`)
- `LGBMClassifier` (if installed: `pip install lightgbm`)
- Soft-voting ensemble of the best individual models

### Hyperparameter tuning
- Use `optuna` to tune any model within a 60-second budget:
  ```python
  import optuna
  optuna.logging.set_verbosity(optuna.logging.WARNING)
  def objective(trial):
      n = trial.suggest_int("n_estimators", 100, 500)
      d = trial.suggest_int("max_depth", 3, 10)
      model = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42)
      return cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()
  study = optuna.create_study(direction="maximize")
  study.optimize(objective, timeout=60)
  best = study.best_params
  ```

### Preprocessing
- `PolynomialFeatures(degree=2, interaction_only=True)` on numeric features
- `SelectKBest` for feature selection

## Rules
- Only modify `train.py`.
- Update `experiment_name` and `notes` for every run.
- The call to `evaluate()`, `log_result()`, and `git_save()` must remain.
- Do not use the test/val set labels to tune — only `X_train`, `y_train`.
- Stop if val_accuracy ≥ 0.85 or after 20 experiments.

## File layout
```
autoresearch/
├── prepare.py   — fixed: data loading, evaluate(), log_result()  ← DO NOT TOUCH
├── train.py     — yours to improve ← ONLY file you edit
├── program.md   — these instructions (human edits this for new directions)
└── results.jsonl — append-only experiment log
```
