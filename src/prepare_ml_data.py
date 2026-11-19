
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

response_path = PROCESSED_DIR / "prism_processed.csv"
features_path = PROCESSED_DIR / "drug_molecular_features.csv"

print("Loading molecular features...")

feature_columns = [
    "column_name",
    "molecular_weight",
    "logp",
    "tpsa",
    "h_bond_donors",
    "h_bond_acceptors",
    "rotatable_bonds",
    "structure_status"
]

drug_features = pd.read_csv(
    features_path,
    usecols=feature_columns
)

print("Loading drug-response data...")

response_columns = [
    "cell_line_id",
    "column_name",
    "broad_id",
    "dose",
    "primary_tissue",
    "logfold_change",
    "failed_str_id"
]

responses = pd.read_csv(
    response_path,
    usecols=response_columns
)

print("Merging datasets...")

df = responses.merge(
    drug_features,
    on="column_name",
    how="left",
    validate="many_to_one"
)

print("Rows before filtering:", len(df))

# Exclude flagged cell lines
df = df[df["failed_str_id"] == False].copy()

# Retain treatments with consistent molecular descriptors
df = df[
    df["structure_status"].isin(
        ["single", "multiple_consistent"]
    )
].copy()

descriptor_columns = [
    "molecular_weight",
    "logp",
    "tpsa",
    "h_bond_donors",
    "h_bond_acceptors",
    "rotatable_bonds"
]

required_columns = (
    descriptor_columns
    + ["dose", "primary_tissue", "logfold_change"]
)

df = df.dropna(subset=required_columns)

# Check for repeated measurements
duplicate_keys = df.duplicated(
    subset=["cell_line_id", "column_name"]
).sum()

print("Duplicate response keys:", duplicate_keys)

if duplicate_keys:
    raise ValueError(
        "Duplicate cell-line/treatment measurements found."
    )

output_columns = [
    "cell_line_id",
    "broad_id",
    "column_name",
    "primary_tissue",
    "dose",
    *descriptor_columns,
    "logfold_change"
]

df = df[output_columns]

output_path = PROCESSED_DIR / "ml_dataset.csv"

df.to_csv(output_path, index=False)

print("\n--- ML DATASET SUMMARY ---")
print("Final shape:", df.shape)
print("Unique compounds:", df["broad_id"].nunique())
print("Unique cell lines:", df["cell_line_id"].nunique())
print("Unique tissues:", df["primary_tissue"].nunique())

print("\nTarget statistics:")
print(df["logfold_change"].describe())

print("\nMissing values:")
print(df.isna().sum())

print("\nSaved to:", output_path)
