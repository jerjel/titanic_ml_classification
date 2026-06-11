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
