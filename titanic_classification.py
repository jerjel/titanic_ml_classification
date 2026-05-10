import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    # ---------------------------------------------------------
    # STEP 1: Load the Data
    # ---------------------------------------------------------
    print("Loading the Titanic dataset...")
    df = pd.read_csv('titanic.csv')
    
    # Display the first few rows to understand the data
    print("\n--- First 5 rows of the dataset ---")
    print(df.head())

    # ---------------------------------------------------------
    # STEP 2: Exploratory Data Analysis & Preprocessing
    # ---------------------------------------------------------
    print("\n--- Data Information ---")
    print(df.info())

    print("\n--- Missing Values Before Cleaning ---")
    print(df.isnull().sum())

    # Handle Missing Values
    # 1. 'Age' has missing values. We'll fill them with the median age.
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    # 2. 'Embarked' has a couple of missing values. We'll fill them with the most common port 'S'.
    df['Embarked'] = df['Embarked'].fillna('S')
    
    # 3. 'Cabin' has too many missing values, so we will drop it. 
    # 'PassengerId', 'Name', and 'Ticket' don't provide useful numeric information for a simple model.
    df.drop(['Cabin', 'PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

    # Encode Categorical Variables
    # Convert 'Sex' (male/female) to 0/1
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    # Convert 'Embarked' (C/Q/S) to dummy variables (One-Hot Encoding)
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

    print("\n--- Missing Values After Cleaning ---")
    print(df.isnull().sum())

    # ---------------------------------------------------------
    # STEP 3: Define Features (X) and Target (y)
    # ---------------------------------------------------------
    # 'Survived' is the column we want to predict
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # ---------------------------------------------------------
    # STEP 4: Split the Data into Training and Testing Sets
    # ---------------------------------------------------------
    # We use 80% of the data for training, and 20% for testing.
    # random_state ensures reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ---------------------------------------------------------
    # STEP 5: Feature Scaling
    # ---------------------------------------------------------
    # Standardize features by removing the mean and scaling to unit variance.
    # This helps many ML models perform better.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ---------------------------------------------------------
    # STEP 6: Build and Train the Model
    # ---------------------------------------------------------
    print("\nTraining the Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # STEP 7: Evaluate the Model
    # ---------------------------------------------------------
    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy * 100:.2f}%\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
