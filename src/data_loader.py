
from pathlib import Path
import pandas as pd

# Project directories
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"

def load_datasets():
    """Load the three PRISM datasets."""

    cell_lines = pd.read_csv(
        RAW_DIR / "primary-screen-cell-line-info.csv"
    )

    drug_response = pd.read_csv(
        RAW_DIR / "primary-screen-replicate-collapsed-logfold-change.csv"
    )

    treatments = pd.read_csv(
        RAW_DIR / "primary-screen-replicate-collapsed-treatment-info.csv"
    )

    return cell_lines, drug_response, treatments


def inspect_datasets():
    cell_lines, drug_response, treatments = load_datasets()

    datasets = {
        "Cell Lines": cell_lines,
        "Drug Response": drug_response,
        "Treatments": treatments
    }

    for name, df in datasets.items():
        print(f"\n--- {name} ---")
        print("Shape:", df.shape)
        print("Missing values:", df.isna().sum().sum())
        print("Duplicate rows:", df.duplicated().sum())

    # Check treatment identifiers
    response_ids = set(drug_response.columns[1:])
    treatment_ids = set(treatments["column_name"].dropna())

    print("\n--- Treatment ID Validation ---")
    print("Matching IDs:", len(response_ids & treatment_ids))
    print("Unmatched response IDs:", len(response_ids - treatment_ids))
    print("Unmatched treatment IDs:", len(treatment_ids - response_ids))

    # Check cell-line identifiers
    response_cell_ids = set(
        drug_response.iloc[:, 0].dropna().astype(str)
    )
    metadata_cell_ids = set(
        cell_lines["depmap_id"].dropna().astype(str)
    )

    print("\n--- Cell-Line ID Validation ---")
    print("Matching IDs:", len(response_cell_ids & metadata_cell_ids))
    print("Unmatched response IDs:", len(response_cell_ids - metadata_cell_ids))


if __name__ == "__main__":
    inspect_datasets()
