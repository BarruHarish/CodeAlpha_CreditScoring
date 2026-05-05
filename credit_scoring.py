# =========================================
# CREDIT SCORING MODEL - FINAL WORKING CODE
# =========================================

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import pickle

# =========================================
# Step 2: Load Dataset
# =========================================
data = pd.read_csv("credit dataset.csv")

print("First 5 rows:")
print(data.head())

print("\nDataset Info:")
print(data.info())

# =========================================
# Step 3: Handle Missing Values
# =========================================
data = data.dropna()

# =========================================
# Step 4: Convert Categorical → Numeric
# =========================================
data = pd.get_dummies(data, drop_first=True)

# =========================================
# Step 5: Split Features & Target
# =========================================
X = data.drop("target", axis=1)
y = data["target"]

# =========================================
# Step 6: Train-Test Split
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================
# Step 7: Train Model
# =========================================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# =========================================
# Step 8: Predictions
# =========================================
y_pred = model.predict(X_test)

# =========================================
# Step 9: Evaluation
# =========================================
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =========================================
# Step 10: Feature Importance
# =========================================
importances = model.feature_importances_
features = X.columns

feat_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(feat_df)

plt.figure(figsize=(8,5))
sns.barplot(x='Importance', y='Feature', data=feat_df)
plt.title("Feature Importance")
plt.show()

# =========================================
# Step 11: Save Model
# =========================================
pickle.dump(model, open("credit_model.pkl", "wb"))

print("\n✅ Model saved as credit_model.pkl")
