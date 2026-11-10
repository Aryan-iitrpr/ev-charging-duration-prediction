# ============================================================
# EV CHARGING DURATION PREDICTION
# AI101 MINI PROJECT - LINEAR REGRESSION
# ============================================================
# 1. IMPORT LIBRARIES
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# 2. SET PROJECT PATHS
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "ev_charging_patterns.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# 3. LOAD DATASET
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}\n"
        "Check that ev_charging_patterns.csv is inside the data folder."
    )
df = pd.read_csv(DATA_PATH)
print("\n========== DATASET INFORMATION ==========")
print("Dataset name:", DATA_PATH.name)
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())

# 4. IDENTIFY TARGET COLUMN
target_candidates = [
    col for col in df.columns
    if "charging duration" in col.lower()
]
if not target_candidates:
    raise ValueError("Charging Duration target column not found.")
    
TARGET = target_candidates[0]
print("\nTarget variable:", TARGET)

# 5. DATA CLEANING
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=[TARGET])
print("\n========== DATA CLEANING ==========")
print("Rows remaining after cleaning:", len(df))
print("Missing target values:", df[TARGET].isna().sum())

# 6. STATISTICAL ANALYSIS
print("\n========== STATISTICAL ANALYSIS ==========")
print("\nMean:")
print(df[TARGET].mean())
print("\nMedian:")
print(df[TARGET].median())
print("\nStandard Deviation:")
print(df[TARGET].std())
numeric_data = df.select_dtypes(include=np.number)
print("\nCovariance Matrix:")
print(numeric_data.cov())
print("\nCorrelation Matrix:")
print(numeric_data.corr())

# 7. Graph 1: TARGET DISTRIBUTION GRAPH
plt.figure(figsize=(7, 4))
plt.hist(df[TARGET].dropna(), bins=30, color="skyblue", edgecolor="black")
plt.title("Distribution of EV Charging Duration")
plt.xlabel("Charging Duration (hours)")
plt.ylabel("Frequency")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(output_dir / "duration_distribution.png")
plt.show()


# 8. FEATURE SELECTION
feature_keywords = [
    "vehicle model",
    "battery capacity",
    "charging rate",
    "time of day",
    "day of week",
    "state of charge (start)",
    "temperature",
    "vehicle age",
    "charger type",
    "user type"
]
selected_features = []
for keyword in feature_keywords:
    matches = [
        col for col in df.columns
        if keyword in col.lower()
    ]

    if matches:
        selected_features.append(matches[0])
if not selected_features:
    raise ValueError("No input features were selected.")

X = df[selected_features]
y = df[TARGET]
print("\n========== FEATURE SELECTION ==========")
print("Selected features:")
print(selected_features)
print("\nNumber of selected features:", len(selected_features))

# 9. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()
categorical_features = X.select_dtypes(
    exclude=np.number
).columns.tolist()
print("\nNumerical features:")
print(numeric_features)
print("\nCategorical features:")
print(categorical_features)

# 10. DATA PREPROCESSING
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ]
)

# 11. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
print("\n========== TRAIN-TEST SPLIT ==========")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# 12. BUILD LINEAR REGRESSION MODEL
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

# 13. TRAIN MODEL
model.fit(X_train, y_train)
print("\n========== MODEL TRAINING ==========")
print("Linear Regression model trained successfully.")

# 14. MAKE PREDICTIONS
y_pred = model.predict(X_test)
# 15. MODEL EVALUATION
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("\n========== LINEAR REGRESSION TEST RESULTS ==========")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")

# 16. SAVE EVALUATION METRICS
metrics_df = pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "R2 Score"],
    "Value": [mae, mse, rmse, r2]
})
metrics_df.to_csv(
    OUTPUT_DIR / "evaluation_metrics.csv",
    index=False
)

# 17. Graph 2: ACTUAL VS PREDICTED GRAPH
plt.figure(figsize=(7, 4))
plt.scatter(y_test, y_pred, alpha=0.6, color="blue")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linestyle="--",
    label="Ideal Prediction",
)
plt.title("Actual vs Predicted Charging Duration")
plt.xlabel("Actual Duration (hours)")
plt.ylabel("Predicted Duration (hours)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(output_dir / "actual_vs_predicted.png")
plt.show()

# 18. Graph 3: RESIDUAL PLOT (Prediction Errors)
residuals = y_test - y_pred
plt.figure(figsize=(7, 4))
plt.scatter(y_pred, residuals, alpha=0.6, color="purple")
plt.axhline(y=0, color="red", linestyle="--")
plt.title("Residual Plot (Prediction Errors)")
plt.xlabel("Predicted Duration (hours)")
plt.ylabel("Error (Actual - Predicted)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(output_dir / "residual_plot.png")
plt.show()

# 19. FINAL SUMMARY
print("\n========== PROJECT SUMMARY ==========")
print("Project: EV Charging Duration Prediction")
print("Algorithm: Linear Regression")
print("Target:", TARGET)
print("Input features:", len(selected_features))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("\nGenerated files:")
print("- outputs/duration_distribution.png")
print("- outputs/actual_vs_predicted.png")
print("- outputs/residual_plot.png")
print("- outputs/evaluation_metrics.csv")
print("\nProject execution completed successfully!")

# Logistic Regression Model

from sklearn.model_selection import train_test_split

# Convert regression target into classification target
threshold = y.median()
y_class = (y > threshold).astype(int)
# 0 = Short charging duration
# 1 = Long charging duration
print("========LOGISTIC REGRESSION MODEL========\n")
print("Classification threshold:", threshold)
print("\nClass distribution:")
print(y_class.value_counts())

# Split dataset
X_train, X_test, y_train_class, y_test_class = train_test_split(
    X, y_class,
    test_size=0.2,
    random_state=42,
    stratify=y_class
)
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import clone

# Create Logistic Regression pipeline
logistic_model = Pipeline([
    ("preprocessor", clone(preprocessor)),
    ("classifier", LogisticRegression(max_iter=1000))
]) 
# Train model
logistic_model.fit(X_train, y_train_class)

# Make predictions
y_pred_class = logistic_model.predict(X_test)
print("\nModel training completed!")
print("Predictions:", y_pred_class[:10])

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt

# Calculate evaluation metrics
print("\n=====EVALUATION METRICES=====\n")
accuracy = accuracy_score(y_test_class, y_pred_class)
precision = precision_score(y_test_class, y_pred_class)
recall = recall_score(y_test_class, y_pred_class)
f1 = f1_score(y_test_class, y_pred_class)

print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test_class, y_pred_class))

# Confusion matrix
cm = confusion_matrix(y_test_class, y_pred_class)

plt.figure(figsize=(6, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig("outputs/logistic_confusion_matrix.png")
plt.show()
# ADDITION : CORRELATION HEATMAP
plt.figure(figsize=(11, 8))

sns.heatmap(
    numeric_data.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Numerical Features")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "correlation_heatmap.png"
)

plt.show()
