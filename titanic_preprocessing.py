
# titanic_preprocessing.py

# Step 1: Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load the dataset
df = pd.read_csv("titanic.csv")

# Step 3: Explore basic info
print("Initial Data Info:")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary statistics:\n", df.describe())

# Step 4: Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Cabin'].fillna("Unknown", inplace=True)

# Step 5: Convert categorical features into numerical
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Step 6: Drop non-informative or high-cardinality features
df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Step 7: Normalize/standardize numerical features
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Step 8: Visualize and remove outliers from Fare using IQR method
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
outlier_condition = (df['Fare'] < (Q1 - 1.5 * IQR)) | (df['Fare'] > (Q3 + 1.5 * IQR))
print(f"\nRemoving {outlier_condition.sum()} outliers from 'Fare' column...")
df = df[~outlier_condition]

# Step 9: Split into features and labels
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 10: Train and evaluate logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
