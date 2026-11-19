
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.error")

path = "data/raw/primary-screen-replicate-collapsed-treatment-info.csv"
df = pd.read_csv(path)

missing = 0
invalid = 0
valid = 0
examples = []

for _, row in df.iterrows():
    smiles = row["smiles"]

    if pd.isna(smiles) or not str(smiles).strip():
        missing += 1
        continue

    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        invalid += 1
        if len(examples) < 10:
            examples.append({
                "name": row["name"],
                "smiles": smiles
            })
    else:
        valid += 1

print("\n--- SMILES VALIDATION ---")
print("Total treatments:", len(df))
print("Valid SMILES:", valid)
print("Missing SMILES:", missing)
print("Invalid SMILES:", invalid)

print("\n--- INVALID SMILES EXAMPLES ---")
for example in examples:
    print("\nDrug:", example["name"])
    print("SMILES:", example["smiles"])
