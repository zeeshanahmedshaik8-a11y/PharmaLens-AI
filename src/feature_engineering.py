
from pathlib import Path
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.error")

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

FEATURES = [
    "molecular_weight",
    "logp",
    "tpsa",
    "h_bond_donors",
    "h_bond_acceptors",
    "rotatable_bonds"
]


def get_descriptors(smiles):
    empty = {feature: None for feature in FEATURES}

    if pd.isna(smiles) or not str(smiles).strip():
        return {**empty, "structure_status": "missing"}

    candidates = [
        item.strip()
        for item in str(smiles).split(",")
        if item.strip()
    ]

    valid_molecules = []
    invalid_count = 0

    for candidate in candidates:
        mol = Chem.MolFromSmiles(candidate)

        if mol is None:
            invalid_count += 1
            continue

        valid_molecules.append(mol)

    if not valid_molecules:
        return {**empty, "structure_status": "invalid"}

    # Do not silently discard invalid candidates
    if invalid_count:
        return {
            **empty,
            "structure_status": "partially_invalid"
        }

    descriptor_sets = []

    for mol in valid_molecules:
        values = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol)
        ]

        descriptor_sets.append(
            tuple(round(value, 6) for value in values)
        )

    unique_sets = set(descriptor_sets)

    if len(unique_sets) > 1:
        return {
            **empty,
            "structure_status": "ambiguous_descriptors"
        }

    values = descriptor_sets[0]

    result = dict(zip(FEATURES, values))

    result["structure_status"] = (
        "single" if len(candidates) == 1
        else "multiple_consistent"
    )

    return result


def main():
    print("Loading treatment metadata...")

    treatments = pd.read_csv(
        RAW_DIR /
        "primary-screen-replicate-collapsed-treatment-info.csv"
    )

    print("Calculating molecular descriptors...")

    unique_smiles = (
        treatments[["smiles"]]
        .drop_duplicates()
        .copy()
    )

    descriptors = unique_smiles["smiles"].apply(
        get_descriptors
    )

    descriptor_df = pd.DataFrame(
        descriptors.tolist(),
        index=unique_smiles.index
    )

    unique_smiles = pd.concat(
        [unique_smiles, descriptor_df],
        axis=1
    )

    result = treatments.merge(
        unique_smiles,
        on="smiles",
        how="left",
        validate="many_to_one"
    )

    print("\n--- FEATURE ENGINEERING RESULTS ---")

    print("Total treatments:", len(result))

    print(
        "\nStructure status:\n",
        result["structure_status"].value_counts()
    )

    print(
        "\nTreatments with descriptors:",
        result["molecular_weight"].notna().sum()
    )

    print(
        "\nTreatments requiring review:",
        result["molecular_weight"].isna().sum()
    )

    print("\nDescriptor sample:")

    print(
        result[
            ["name", "molecular_weight",
             "logp", "tpsa", "structure_status"]
        ].head(10).to_string(index=False)
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output = (
        PROCESSED_DIR / "drug_molecular_features.csv"
    )

    result.to_csv(output, index=False)

    print("\nSaved to:", output)


if __name__ == "__main__":
    main()
