
from pathlib import Path
import json

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "processed" / "ml_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

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

print("Loading dataset...")

df = pd.read_csv(DATA_FILE)

tissue_features = pd.get_dummies(
    df["primary_tissue"],
    prefix="tissue",
    dtype=np.int8
)

X = pd.concat(
    [df[NUMERIC_FEATURES], tissue_features],
    axis=1
).astype(np.float32)

groups = df["broad_id"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, groups=groups)
)

X_test = X.iloc[test_idx]

# Restore exact training feature order
with open(
    MODEL_DIR / "xgboost_features.json"
) as file:
    feature_names = json.load(file)

X_test = X_test.reindex(
    columns=feature_names,
    fill_value=0
)

print("Loading trained XGBoost model...")

model = XGBRegressor()
model.load_model(
    str(MODEL_DIR / "xgboost_model.json")
)

# Reproducible test sample
X_sample = X_test.sample(
    n=min(2000, len(X_test)),
    random_state=42
)

print("Calculating SHAP values...")

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_sample)

print("Generating SHAP summary plot...")

shap.summary_plot(
    shap_values.values,
    X_sample,
    show=False,
    max_display=15
)

plt.tight_layout()
plt.savefig(
    REPORT_DIR / "shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Generating SHAP feature importance...")

shap.summary_plot(
    shap_values.values,
    X_sample,
    plot_type="bar",
    show=False,
    max_display=15
)

plt.tight_layout()
plt.savefig(
    REPORT_DIR / "shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# Save numerical feature importance
importance = pd.DataFrame({
    "feature": X_sample.columns,
    "mean_abs_shap": np.abs(
        shap_values.values
    ).mean(axis=0)
})

importance = importance.sort_values(
    "mean_abs_shap",
    ascending=False
)

importance.to_csv(
    REPORT_DIR / "shap_feature_importance.csv",
    index=False
)

print("\n--- TOP 10 IMPORTANT FEATURES ---")
print(importance.head(10).to_string(index=False))

print("\nSHAP analysis completed!")
print("Plots saved in:", REPORT_DIR)
