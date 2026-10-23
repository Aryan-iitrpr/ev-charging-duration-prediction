from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")

# Project folders
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Find the dataset CSV
csv_files = list(DATA_DIR.glob("*.csv")) if DATA_DIR.exists() else []
if not csv_files:
    csv_files = [
        p for p in ROOT.glob("*.csv")
        if p.name.lower() not in ("results.csv",)
    ]

if not csv_files:
    raise FileNotFoundError(
        "Dataset not found. Download the EV charging CSV "
        "and place it inside the data folder."
    )

DATA_PATH = csv_files[0]
df = pd.read_csv(DATA_PATH)

print("=" * 55)
print("EV CHARGING DURATION PREDICTION")
print("=" * 55)
print(f"\nDataset: {DATA_PATH.name}")
print(f"Dataset shape: {df.shape}")
print("\nOriginal columns:")
print(df.columns.tolist())

# Normalize column names for reliable matching
def normalize(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")

column_map = {normalize(c): c for c in df.columns}

target_candidates = [
    "charging_duration_hours",
    "charging_duration",
    "chargingduration",
    "duration_hours",
]

target_col = next(
    (column_map[c] for c in target_candidates if c in column_map),
    None
)

if target_col is None:
    target_col = next(
        (
            c for c in df.columns
            if "duration" in normalize(c)
            and "charging" in normalize(c)
        ),
        None
    )

if target_col is None:
    raise ValueError(
        "Could not identify the charging-duration column. "
        "Check the printed column names and update target_col."
    )

# Convert target to numeric and remove invalid target rows
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
df = df.dropna(subset=[target_col]).copy()
df = df[df[target_col] >= 0].copy()

print(f"\nTarget column: {target_col}")
print(f"Rows after target cleaning: {len(df)}")

# Basic statistical analysis
numeric_df = df.select_dtypes(include=np.number)

print("\nNUMERICAL SUMMARY")
print(numeric_df.describe().round(3))

print("\nMean:")
print(numeric_df.mean().round(3))

print("\nMedian:")
print(numeric_df.median().round(3))

print("\nStandard deviation:")
print(numeric_df.std().round(3))

print("\nCovariance matrix:")
print(numeric_df.cov().round(3))

print("\nCorrelation matrix:")
print(numeric_df.corr().round(3))

# Plot target distribution
plt.figure(figsize=(8, 5))
plt.hist(df[target_col], bins=25, edgecolor="black")
plt.xlabel("Charging duration (hours)")
plt.ylabel("Number of sessions")
plt.title("Distribution of EV Charging Duration")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "duration_distribution.png", dpi=150)
plt.close()

# Select features available before or at the start of charging.
# Avoid IDs, end timestamps, energy delivered, cost, and other
# variables that can reveal information from the completed session.
feature_keywords = [
    "battery_capacity",
    "charging_rate",
    "state_of_charge_start",
    "temperature",
    "vehicle_age",
    "vehicle_model",
    "charger_type",
    "user_type",
    "time_of_day",
    "day_of_week",
]

feature_cols = [
    c for c in df.columns
    if c != target_col
    and any(k in normalize(c) for k in feature_keywords)
]

if not feature_cols:
    raise ValueError(
        "No expected predictor columns were found. "
        "Inspect the printed column names and update feature_cols."
    )

# Remove columns that contain no usable values
feature_cols = [
    c for c in feature_cols
    if df[c].notna().any()
]

X = df[feature_cols].copy()
y = df[target_col].copy()

# Convert numeric-looking columns where appropriate
for col in X.columns:
    if X[col].dtype == "object":
        converted = pd.to_numeric(X[col], errors="coerce")
        if converted.notna().mean() > 0.8:
            X[col] = converted

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = [
    c for c in X.columns if c not in numeric_features
]

print("\nSelected features:")
print(feature_cols)
print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)

# Train/test split: keep test data separate from model training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression()),
])

# Train the model
model.fit(X_train, y_train)

# Evaluate on training and unseen test data
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

def evaluate(label, actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return {
        "Dataset": label,
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(actual, predicted),
    }

results = pd.DataFrame([
    evaluate("Training", y_train, train_pred),
    evaluate("Testing", y_test, test_pred),
])

print("\nMODEL EVALUATION")
print(results.round(4).to_string(index=False))

results.to_csv(OUTPUT_DIR / "evaluation_metrics.csv", index=False)

# Actual vs predicted plot
plt.figure(figsize=(7, 6))
plt.scatter(y_test, test_pred, alpha=0.7)

lower = min(y_test.min(), test_pred.min())
upper = max(y_test.max(), test_pred.max())
plt.plot([lower, upper], [lower, upper], "r--")

plt.xlabel("Actual charging duration (hours)")
plt.ylabel("Predicted charging duration (hours)")
plt.title("Actual vs Predicted: Test Set")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=150)
plt.close()

# Residual plot
residuals = y_test.to_numpy() - test_pred
plt.figure(figsize=(7, 5))
plt.scatter(test_pred, residuals, alpha=0.7)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted charging duration (hours)")
plt.ylabel("Residual (actual - predicted)")
plt.title("Residual Plot: Test Set")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "residual_plot.png", dpi=150)
plt.close()

# Save example predictions
prediction_table = pd.DataFrame({
    "Actual_duration_hours": y_test.to_numpy(),
    "Predicted_duration_hours": test_pred,
})
prediction_table.to_csv(
    OUTPUT_DIR / "sample_predictions.csv", index=False
)

print("\nSaved results and plots in the outputs folder.")
print("Review the test metrics before drawing conclusions.")
