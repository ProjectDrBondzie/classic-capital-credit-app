import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score

DATA_PATH = Path("classic_capital_credit_dataset_filled_sample.csv")
MODEL_PATH = Path("credit_model.pkl")

# ---------------------------
# Load data
# ---------------------------
df = pd.read_csv(DATA_PATH)

# ---------------------------
# Target
# ---------------------------
target = "default_flag"

# ---------------------------
# Drop columns not used in training
# ---------------------------
drop_cols = [
    "borrower_name",
    "member_id",
    "application_date",
    "loan_id",
    "loan_status",
    "amount_repaid",
    "days_past_due"
]

df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

# ---------------------------
# Explicit feature groups
# ---------------------------
numeric_features = [
    "age",
    "household_size",
    "years_employed",
    "monthly_income",
    "monthly_expenses",
    "monthly_savings",
    "other_debt_amount",
    "existing_loans",
    "collateral_value",
    "member_years",
    "loan_amount",
    "loan_term_months",
    "interest_rate",
]

categorical_features = [
    "gender",
    "marital_status",
    "education_level",
    "employment_type",
    "business_owner",
    "guarantor",
    "loan_purpose",
    "repayment_frequency",
]

# ---------------------------
# Keep only columns that exist
# ---------------------------
numeric_features = [col for col in numeric_features if col in df.columns]
categorical_features = [col for col in categorical_features if col in df.columns]

# ---------------------------
# Convert numeric columns safely
# ---------------------------
for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------------------------
# Convert categorical columns safely
# ---------------------------
for col in categorical_features:
    df[col] = df[col].fillna("Unknown").astype(object)

# ---------------------------
# Remove feature columns that are entirely missing
# ---------------------------
numeric_features = [col for col in numeric_features if df[col].notna().sum() > 0]
categorical_features = [col for col in categorical_features if df[col].notna().sum() > 0]

# ---------------------------
# Clean target
# ---------------------------
df[target] = pd.to_numeric(df[target], errors="coerce")
df = df[df[target].notna()].copy()
df[target] = df[target].astype(int)

# ---------------------------
# Build X and y
# ---------------------------
feature_cols = numeric_features + categorical_features
X = df[feature_cols].copy()
y = df[target].copy()

# ---------------------------
# Preprocessing
# ---------------------------
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ---------------------------
# Model pipeline
# ---------------------------
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

# ---------------------------
# Train test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# ---------------------------
# Train model
# ---------------------------
model.fit(X_train, y_train)

# ---------------------------
# Predictions
# ---------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ---------------------------
# Evaluation
# ---------------------------
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\nFeatures used in training:")
print(feature_cols)

print("\nModel Evaluation")
print("-" * 40)
print(f"Accuracy: {acc:.4f}")
print(f"AUC:      {auc:.4f}")
print("\nConfusion Matrix")
print(cm)
print("\nClassification Report")
print(report)

# ---------------------------
# Save model
# ---------------------------
joblib.dump(model, MODEL_PATH)
print(f"\nSaved trained model to: {MODEL_PATH}")