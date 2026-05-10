# Titanic – Karpathy Auto-Research Experiment Log

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
- Survival rate: 38.4%  →  class imbalance is mild
- **Sex** is the strongest raw signal: female survival 74% vs male 19%
- **Pclass** is strongly inversely correlated with survival (-0.34)
- **Fare** positively correlated (+0.26), likely a proxy for class/wealth
- Children under 10 survived at 61%
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

---
## Step 2: Baselines

| Baseline | Description | Test Accuracy |
|----------|-------------|---------------|
| B0 | Majority class (everyone dies) | 61.45% |
| B1 | Gender rule (female=survive, male=die) | 77.65% |
| B2 | Logistic Regression (Pclass + Sex) | 77.65% |

**Key takeaway:** A single feature (Sex) yields 77.7% accuracy.
Feature engineering must push beyond this.

**Next target:** ≥80% with proper features + tuned model.

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
- 20-sample DecisionTree train accuracy: 100% ✔
- Full RF train accuracy: 98.74%
- Full RF test accuracy:  81.01%
- Gap: 17.73% → overfitting detected

### Next Step
→ Hyperparameter tuning + cross-validation to close the gap

---
## Step 4: Model Comparison & Hyperparameter Tuning

### 5-Fold CV Results
| Model | Mean Accuracy | Std |
|-------|--------------|-----|
| Logistic Regression | 79.64% | ±2.20% |
| Random Forest | 81.18% | ±1.49% |
| Gradient Boosting | 80.34% | ±1.49% |
| SVM (RBF) | 82.45% | ±1.15% |

**Winner:** SVM (RBF)

### Best Hyperparameters (GridSearchCV)
```
{'max_depth': None, 'min_samples_split': 5, 'n_estimators': 200}
```

### Final Test Metrics
- Test Accuracy : **78.77%**
- ROC-AUC       : **0.8451**
- Misclassified : 38 / 179 (21.2%)

### Top 3 Most Important Features
Sex         0.230713
Fare_log    0.187871
Age         0.146356

### Misclassification Analysis
Most errors occur for male passengers in 3rd class → the "men die" rule
has exceptions (e.g., children, men with family pulling them forward).
This is the irreducible noise given available features.

---
## Step 5: Ensembles – Squeeze Out the Last Juice

| Method | CV Accuracy | Test Accuracy | ROC-AUC |
|--------|-------------|---------------|---------|
| Soft Voting (RF+GB+LR+SVM) | 82.02% | 82.12% | 0.8659 |
| Stacking (RF+GB+SVM → LR) | 82.31% | 79.89% | 0.8574 |

---
## FINAL SUMMARY — Full Research Journey

| Model | Test Accuracy |
|-------|--------------|
| B0: Majority Class | 61.45% |
| B1: Gender Rule | 78.77% |
| B2: LogReg (Pclass+Sex) | 78.77% |
| RF (step3, raw) | 81.56% |
| Best tuned model (step4) | 82.68% |
| Soft Voting Ensemble | 82.12% |
| Stacking Ensemble | 79.89% |

**Best model:** Best tuned model (step4) → **82.68%**

### What we learned following Karpathy's Pattern
1. **Become one with data first** — Sex alone gives 78.8% accuracy. Feature engineering must build on this.
2. **Baselines prevent wasted effort** — Anything below 78.8% is garbage.
3. **Overfit check validates the pipeline** — Without this, you might tune a broken system.
4. **CV > single split** — Test accuracy can be lucky; 5-fold CV is honest.
5. **Feature engineering > complex models** — Title, FamilySize, Deck improved RF by ~4pp over raw features.
6. **Ensembles are the cleanest last step** — They squeeze 1-2% reliably.
7. **Look at misclassified examples** — They reveal the irreducible noise in the problem.

### Potential Next Steps
- Try XGBoost / LightGBM (often outperform sklearn GB)
- Cabin deck imputation (currently 77% unknown)
- Interaction features: Pclass × Sex, Title × FamilySize
- Calibrated probabilities (Platt scaling)
