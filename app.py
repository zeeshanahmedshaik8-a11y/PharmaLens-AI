
from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

st.set_page_config(
    page_title="PharmaLens AI",
    page_icon="💊",
    layout="wide"
)

NUMERIC_FEATURES = [
    "dose",
    "molecular_weight",
    "logp",
    "tpsa",
    "h_bond_donors",
    "h_bond_acceptors",
    "rotatable_bonds"
]


@st.cache_resource
def load_model():
    model = XGBRegressor()
    model.load_model(
        str(MODEL_DIR / "xgboost_model.json")
    )

    with open(
        MODEL_DIR / "xgboost_features.json"
    ) as file:
        feature_names = json.load(file)

    return model, feature_names


@st.cache_data
def load_results():
    with open(
        REPORT_DIR / "xgboost_results.json"
    ) as file:
        return json.load(file)


model, feature_names = load_model()
results = load_results()

st.title("💊 PharmaLens AI")
st.caption(
    "Explainable Machine Learning for Experimental "
    "Drug Response Prediction"
)

st.warning(
    "Research demonstration only. Predictions represent "
    "estimated in-vitro assay responses, not clinical "
    "treatment recommendations."
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Project Overview",
        "Drug Response Prediction",
        "Explainable AI"
    ]
)

if page == "Project Overview":

    st.header("Project Overview")

    st.write(
        "PharmaLens AI analyzes PRISM drug-response data "
        "using molecular descriptors, cancer tissue "
        "information, and XGBoost."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "MAE",
        f"{results['mae']:.4f}"
    )

    col2.metric(
        "RMSE",
        f"{results['rmse']:.4f}"
    )

    col3.metric(
        "R² Score",
        f"{results['r2']:.4f}"
    )

    st.subheader("Dataset")

    st.write(
        f"Training measurements: "
        f"{results['train_records']:,}"
    )

    st.write(
        f"Testing measurements: "
        f"{results['test_records']:,}"
    )

    st.write(
        f"Training compounds: "
        f"{results['train_compounds']:,}"
    )

    st.write(
        f"Testing compounds: "
        f"{results['test_compounds']:,}"
    )

    st.info(
        "Evaluation uses a compound-grouped train/test "
        "split. Predictive performance on unseen compounds "
        "is currently limited."
    )


elif page == "Drug Response Prediction":

    st.header("Predict Experimental Drug Response")

    st.write(
        "Enter molecular descriptors and experimental "
        "conditions to estimate log-fold change."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            dose = st.number_input(
                "Dose (dataset units)",
                min_value=0.0,
                value=2.5
            )

            molecular_weight = st.number_input(
                "Molecular Weight",
                min_value=0.0,
                value=350.0
            )

            logp = st.number_input(
                "LogP",
                value=2.0
            )

            tpsa = st.number_input(
                "TPSA",
                min_value=0.0,
                value=80.0
            )

        with col2:
            h_bond_donors = st.number_input(
                "Hydrogen Bond Donors",
                min_value=0,
                value=2
            )

            h_bond_acceptors = st.number_input(
                "Hydrogen Bond Acceptors",
                min_value=0,
                value=5
            )

            rotatable_bonds = st.number_input(
                "Rotatable Bonds",
                min_value=0,
                value=4
            )

            tissue_columns = [
                name.replace("tissue_", "", 1)
                for name in feature_names
                if name.startswith("tissue_")
            ]

            tissue = st.selectbox(
                "Cancer Tissue",
                sorted(tissue_columns)
            )

        submitted = st.form_submit_button(
            "Predict Drug Response"
        )

    if submitted:

        row = {
            feature: 0.0
            for feature in feature_names
        }

        row.update({
            "dose": dose,
            "molecular_weight": molecular_weight,
            "logp": logp,
            "tpsa": tpsa,
            "h_bond_donors": h_bond_donors,
            "h_bond_acceptors": h_bond_acceptors,
            "rotatable_bonds": rotatable_bonds
        })

        row[f"tissue_{tissue}"] = 1.0

        input_df = pd.DataFrame(
            [row],
            columns=feature_names
        ).astype(np.float32)

        prediction = model.predict(input_df)[0]

        st.metric(
            "Predicted Log-Fold Change",
            f"{prediction:.4f}"
        )

        st.caption(
            "Negative values indicate lower measured "
            "viability relative to the assay reference. "
            "This prediction is not a clinical outcome."
        )


elif page == "Explainable AI":

    st.header("SHAP Model Explainability")

    st.write(
        "SHAP estimates how individual features "
        "contribute to the model's predictions."
    )

    summary_path = REPORT_DIR / "shap_summary.png"
    importance_path = (
        REPORT_DIR / "shap_feature_importance.png"
    )

    if summary_path.exists():
        st.subheader("SHAP Summary")
        st.image(str(summary_path), width="stretch")

    if importance_path.exists():
        st.subheader("Feature Importance")
        st.image(str(importance_path), width="stretch")

    csv_path = (
        REPORT_DIR / "shap_feature_importance.csv"
    )

    if csv_path.exists():
        importance = pd.read_csv(csv_path)
        st.dataframe(
            importance.head(15),
            hide_index=True
        )

    st.info(
        "SHAP values explain model predictions, "
        "not causal biological mechanisms."
    )

st.divider()
st.caption(
    "PharmaLens AI | Pharmaceutical Data Science Project"
)
