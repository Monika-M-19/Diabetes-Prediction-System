import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

data = pd.read_csv("diabetes.csv")


# --------------------------------------------------
# 2. Separate features and target
# --------------------------------------------------

X = data.drop("Outcome", axis=1)
y = data["Outcome"]


# --------------------------------------------------
# 3. Replace invalid zero values with NaN
# --------------------------------------------------

columns_to_clean = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

X[columns_to_clean] = X[columns_to_clean].replace(0, np.nan)


# --------------------------------------------------
# 4. Split data into training and testing
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 5. Fill missing values using training median
# --------------------------------------------------

imputer = SimpleImputer(strategy="median")

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)


# --------------------------------------------------
# 6. Feature scaling
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)


# --------------------------------------------------
# 7. Create and train Logistic Regression model
# --------------------------------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)


# --------------------------------------------------
# 8. Make predictions
# --------------------------------------------------

y_pred = model.predict(X_test_scaled)


# --------------------------------------------------
# 9. Evaluate model
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)


# --------------------------------------------------
# 10. Save confusion matrix
# --------------------------------------------------

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Diabetes", "Diabetes"],
    yticklabels=["No Diabetes", "Diabetes"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Diabetes Prediction")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=160, bbox_inches="tight")
plt.close()


# --------------------------------------------------
# 11. Save model, scaler and imputer
# --------------------------------------------------

joblib.dump(model, "diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(imputer, "imputer.pkl")

print("\nModel, scaler, imputer and confusion matrix saved successfully!")
