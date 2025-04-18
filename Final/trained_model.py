

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
file_path = "balanced_dataset.csv"
df = pd.read_csv(file_path)

# Drop missing values (if any)
df.dropna(inplace=True)



# Separate fraud and non-fraud cases
fraud_df = df[df['FraudFound_P'] == 1]
non_fraud_df = df[df['FraudFound_P'] == 0]

# Under-sample non-fraud to match fraud cases
non_fraud_sample = non_fraud_df.sample(n=len(fraud_df), random_state=42)

# Combine to create a balanced dataset
balanced_df = pd.concat([fraud_df, non_fraud_sample])

# Shuffle the dataset
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Balanced dataset shape: {balanced_df.shape}")
print(balanced_df['FraudFound_P'].value_counts())



label_encoders = {}
categorical_columns = [
    "Month", "DayOfWeek", "AccidentArea", "DayOfWeekClaimed",
    "MonthClaimed", "PastNumberOfClaims", "AgeOfVehicle",
    "PoliceReportFiled", "WitnessPresent"
]

for col in categorical_columns:
    le = LabelEncoder()
    balanced_df[col] = le.fit_transform(balanced_df[col])
    label_encoders[col] = le  # Save encoder for future use

#  Define features and target

X = balanced_df.drop(columns=["FraudFound_P"])
y = balanced_df["FraudFound_P"]

#  Scale features

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

#  Train Gradient Boosting Classifier

model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

#  Evaluate model

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

#  Save model and preprocessing objects

joblib.dump(model, "fraud_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "encoders.pkl")

print("✅ Model and preprocessing tools saved successfully!")


