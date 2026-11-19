
from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from xgboost import XGBRegressor

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

print("Loading ML dataset...")

df = pd.read_csv(DATA_FILE)

# Encode tissue categories consistently
tissue_features = pd.get_dummies(
    df["primary_tissue"],
    prefix="tissue",
    dtype=np.int8
)

X = pd.concat(
    [df[NUMERIC_FEATURES], tissue_features],
    axis=1
).astype(np.float32)

y = df["logfold_change"]
groups = df["broad_id"]

# Same compound-grouped split as Ridge
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

print("\nTraining XGBoost...")

model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    tree_method="hist",
    n_jobs=4,
    random_state=42
)

model.fit(X_train, y_train)

print("Generating predictions...")

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n--- XGBOOST RESULTS ---")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")

# Save model
model_path = MODEL_DIR / "xgboost_model.json"
model.save_model(str(model_path))

# Save exact feature order for future inference
feature_path = MODEL_DIR / "xgboost_features.json"

with open(feature_path, "w") as file:
    json.dump(list(X.columns), file, indent=4)

results = {
    "model": "XGBoost",
    "split": "Compound-grouped 80/20",
    "random_state": 42,
    "train_records": len(X_train),
    "test_records": len(X_test),
    "train_compounds": len(train_compounds),
    "test_compounds": len(test_compounds),
    "mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2)
}

with open(
    REPORT_DIR / "xgboost_results.json",
    "w"
) as file:
    json.dump(results, file, indent=4)

print("\nModel saved to:", model_path)
print("Results saved to reports/xgboost_results.json")
print("XGBoost training completed!")
