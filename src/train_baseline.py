
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "processed" / "ml_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

NUMERIC_FEATURES = [
    "dose",
    "molecular_weight",
    "logp",
    "tpsa",
    "h_bond_donors",
    "h_bond_acceptors",
    "rotatable_bonds"
]

CATEGORICAL_FEATURES = ["primary_tissue"]

print("Loading ML dataset...")

df = pd.read_csv(DATA_FILE)

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df["logfold_change"]
groups = df["broad_id"]

# Keep all measurements from each compound together
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

train_compounds = set(groups.iloc[train_idx])
test_compounds = set(groups.iloc[test_idx])

assert train_compounds.isdisjoint(test_compounds)

print("\n--- DATA SPLIT ---")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Training compounds:", len(train_compounds))
print("Testing compounds:", len(test_compounds))
print("Compound overlap:", len(train_compounds & test_compounds))

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore"
)

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, NUMERIC_FEATURES),
    ("cat", categorical_transformer, CATEGORICAL_FEATURES)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", Ridge(alpha=10.0))
])

print("\nTraining Ridge Regression baseline...")
model.fit(X_train, y_train)

print("Generating predictions...")
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)
r2 = r2_score(y_test, predictions)

# Compare against a constant training-mean predictor
baseline_predictions = np.full(
    len(y_test),
    y_train.mean()
)

baseline_mae = mean_absolute_error(
    y_test, baseline_predictions
)
baseline_rmse = np.sqrt(
    mean_squared_error(y_test, baseline_predictions)
)

print("\n--- MODEL EVALUATION ---")
print(f"Mean baseline MAE: {baseline_mae:.4f}")
print(f"Mean baseline RMSE: {baseline_rmse:.4f}")
print(f"Ridge MAE: {mae:.4f}")
print(f"Ridge RMSE: {rmse:.4f}")
print(f"Ridge R2: {r2:.4f}")

results = {
    "model": "Ridge Regression",
    "split": "Compound-grouped 80/20",
    "random_state": 42,
    "train_records": len(X_train),
    "test_records": len(X_test),
    "train_compounds": len(train_compounds),
    "test_compounds": len(test_compounds),
    "mean_baseline_mae": baseline_mae,
    "mean_baseline_rmse": baseline_rmse,
    "ridge_mae": mae,
    "ridge_rmse": rmse,
    "ridge_r2": r2
}

with open(
    REPORT_DIR / "baseline_results.json",
    "w"
) as file:
    json.dump(results, file, indent=4)

print("\nResults saved to reports/baseline_results.json")
print("Baseline training completed!")
