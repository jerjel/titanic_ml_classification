# Titanic ML Classification: A Step-by-Step Learning Journey

This project is a comprehensive guide to solving the classic Titanic survival prediction problem using Machine Learning. It covers everything from initial data exploration (EDA) to advanced model ensembling, following an iterative "Auto-Research" pattern.

---

## 🚀 Project Overview

The goal is to predict which passengers survived the Titanic shipwreck based on features like age, sex, passenger class, and fare.

### Repository Structure
- `titanic.csv`: The raw dataset.
- `data_exploration.py`: Script for deep Exploratory Data Analysis (EDA).
- `titanic_classification.py`: A standard baseline classification script.
- `plots/`: Visualisations generated during EDA.
- `karpathy_approach/`: A structured sub-project following Andrej Karpathy's iterative research methodology (see its internal README for details).

---

## 📋 The Machine Learning Process

### 1. Exploratory Data Analysis (EDA)
**What was done:** 
We used `pandas`, `matplotlib`, and `seaborn` to visualize every column in the dataset. We looked at survival rates across different categories (Sex, Pclass, Port of Embarkation) and analyzed distributions of continuous variables (Age, Fare).

**Why it was done:**
- **Understand the "Signal":** We discovered that `Sex` is the strongest predictor (females had a ~74% survival rate vs ~19% for males).
- **Identify Data Issues:** We found that `Cabin` is 77% missing and `Age` is 20% missing. This tells us we need a strategy for handling "holes" in our data.
- **Spot Anomalies:** We identified a "long tail" in the `Fare` data, suggesting that a log-transformation might help the model process the values better.

### 2. Data Preprocessing & Feature Engineering
**What was done:**
- **Imputation:** Filling missing `Age` values using the median.
- **Categorical Encoding:** Converting "male/female" into 0/1 so the computer can understand it.
- **Feature Creation:** Extracting `Title` (Mr, Mrs, Master) from names and calculating `FamilySize` (Siblings + Parents).

**Why it was done:**
- **Mathematical Compatibility:** Machine Learning models (like Random Forests or Logistic Regression) only speak "numbers," not text.
- **Boosting Signal:** A raw name is useless to a model, but a title like "Master" tells the model the passenger is a young boy, who was more likely to be prioritized for lifeboats.

### 3. The "Karpathy" Research Pattern
**What was done:**
In the `karpathy_approach/` folder, we broke the problem down into 5 strict stages:
1. **Become one with the data:** Deep inspection.
2. **Set a dumb baseline:** Majority class and "Gender Rule" (Women survive/Men die).
3. **Overfit a tiny batch:** Proving the code can "memorize" 20 people to ensure the pipeline isn't broken.
4. **Regularize & Tune:** Using Cross-Validation and GridSearch to find the best settings.
5. **Ensemble:** Combining multiple different models (SVM, Gradient Boosting, RF) to get the final 1-2% accuracy boost.

**Why it was done:**
This prevents "hero coding"—jumping straight to complex models without verifying the basics. It ensures every bit of complexity is justified by a real increase in accuracy.

---

## 💡 Key Machine Learning Concepts Used

- **Overfitting:** When a model learns the "noise" of the training data too well and fails on new, unseen data. We used **Cross-Validation** to prevent this.
- **Feature Importance:** A way to see which variables (like Sex or Pclass) the model relied on most for its decisions.
- **Hyperparameter Tuning:** Adjusting the "knobs" of the model (like the depth of a tree) to find the sweet spot between simplicity and accuracy.
- **Ensembling:** The "Wisdom of the Crowd." Combining different models to cancel out individual errors.

---

## 🛠️ How to Use This Project

### 1. Prerequisites
Ensure you have Python installed with the necessary libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 2. Run Data Exploration
See the patterns in the data yourself:
```bash
python data_exploration.py
```

### 3. Run the Advanced Iterative Pipeline
Follow the Karpathy process step-by-step:
```bash
cd karpathy_approach
python 01_become_one_with_data.py
python 02_set_baselines.py
# ... and so on
```

---

## 📊 Final Results
By following this process, we moved from a "blind guess" (61.5%) to a sophisticated ensemble model achieving **~82-84% accuracy**.

**Created by Jerry Jacob**
*Iterative Machine Learning Research*
