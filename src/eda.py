
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "processed" / "prism_processed.csv"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)

print("Loading processed dataset...")

# Load only columns needed for initial EDA
df = pd.read_csv(
    DATA_FILE,
    usecols=[
        "cell_line_id",
        "column_name",
        "logfold_change",
        "primary_tissue",
        "failed_str_id"
    ]
)

print("Original dataset shape:", df.shape)

# Exclude flagged cell lines for initial analysis
df = df[df["failed_str_id"] == False].copy()

print("Quality-filtered dataset shape:", df.shape)

print("\n--- Drug Response Statistics ---")
print(df["logfold_change"].describe())

print("\n--- Missing Values ---")
print(df.isna().sum())

print("\n--- Top 10 Tissue Types ---")
print(df["primary_tissue"].value_counts().head(10))

# Plot 1: Drug-response distribution
plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="logfold_change",
    bins=100
)

plt.title("PRISM Drug Response Distribution")
plt.xlabel("Log Fold Change")
plt.ylabel("Number of Measurements")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "drug_response_distribution.png",
    dpi=300
)
plt.close()

# Plot 2: Top 10 tissue types
plt.figure(figsize=(11, 6))

tissue_counts = (
    df["primary_tissue"]
    .value_counts()
    .head(10)
)

sns.barplot(
    x=tissue_counts.values,
    y=tissue_counts.index
)

plt.title("Top 10 Cancer Tissue Types")
plt.xlabel("Number of Drug-Response Measurements")
plt.ylabel("Primary Tissue")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "tissue_distribution.png",
    dpi=300
)
plt.close()

print("\nEDA completed successfully!")
print("Plots saved in:", REPORT_DIR)
