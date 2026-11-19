
from pathlib import Path
import pandas as pd

from data_loader import load_datasets

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def preprocess_data():
    print("Loading PRISM datasets...")
    cell_lines, response, treatments = load_datasets()

    # Rename the response matrix identifier
    response = response.rename(
        columns={"Unnamed: 0": "cell_line_id"}
    )

    # Convert wide matrix to long format
    print("Converting response matrix to long format...")

    response_long = response.melt(
        id_vars="cell_line_id",
        var_name="column_name",
        value_name="logfold_change"
    )

    print("Rows before removing missing responses:",
          len(response_long))

    # Missing measurements are not equivalent to zero
    response_long = response_long.dropna(
        subset=["logfold_change"]
    )

    print("Rows after removing missing responses:",
          len(response_long))

    # Merge treatment metadata
    print("Merging treatment information...")

    merged = response_long.merge(
        treatments,
        on="column_name",
        how="left",
        validate="many_to_one",
        indicator=True
    )

    print("Treatment merge results:")
    print(merged["_merge"].value_counts())
    merged = merged.drop(columns=["_merge"])

    # Merge cell-line metadata using row_name
    print("Merging cell-line information...")

    merged = merged.merge(
        cell_lines,
        left_on="cell_line_id",
        right_on="row_name",
        how="left",
        validate="many_to_one",
        indicator=True
    )

    print("Cell-line merge results:")
    print(merged["_merge"].value_counts())
    merged = merged.drop(columns=["_merge"])

    # Preserve the original STR quality information
    merged["failed_str_id"] = (
        merged["cell_line_id"]
        .astype(str)
        .str.endswith("_FAILED_STR")
    )

    # Save processed data
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = PROCESSED_DIR / "prism_processed.csv"

    merged.to_csv(output_file, index=False)

    print("\n--- PREPROCESSING COMPLETE ---")
    print("Final shape:", merged.shape)
    print("Unique cell lines:",
          merged["cell_line_id"].nunique())
    print("Unique treatments:",
          merged["column_name"].nunique())
    print("Rows with failed STR identifiers:",
          merged["failed_str_id"].sum())
    print("Saved to:", output_file)


if __name__ == "__main__":
    preprocess_data()
