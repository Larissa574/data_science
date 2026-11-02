"""Streamlit application for FinAccess AI."""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from predict import load_model, predict  # noqa: E402

MODEL_PATH = ROOT / "models" / "model.joblib"

st.set_page_config(page_title="FinAccess AI", page_icon="🏦", layout="wide")
st.title("FinAccess AI")
st.caption("Prédiction de l'accès aux services bancaires à partir d'un profil socio-économique")

if not MODEL_PATH.exists():
    st.warning("Aucun modèle entraîné n'est disponible. Lancez d'abord src/train.py après avoir placé le dataset dans data/.")
    st.stop()

model = load_model(MODEL_PATH)

st.sidebar.header("Navigation")
page = st.sidebar.radio("Choisir une page", ["Prédiction", "Prédiction CSV", "À propos"])

if page == "Prédiction":
    st.subheader("Prédiction individuelle")
    st.info("Les champs affichés sont dérivés des variables attendues par le modèle.")

    try:
        feature_names = list(model.feature_names_in_)
    except AttributeError:
        feature_names = []

    if not feature_names:
        st.error("Impossible de récupérer les variables attendues par le modèle.")
        st.stop()

    values = {}
    cols = st.columns(2)
    for i, feature in enumerate(feature_names):
        with cols[i % 2]:
            values[feature] = st.text_input(feature, key=f"feature_{i}")

    if st.button("Prédire", type="primary"):
        row = pd.DataFrame([values])
        result = predict(row, MODEL_PATH)
        st.success(f"Classe prédite : {result.loc[0, 'prediction']}")
        if "probability" in result:
            st.metric("Probabilité de la classe prédite", f"{result.loc[0, 'probability']:.2%}")

elif page == "Prédiction CSV":
    st.subheader("Prédiction par fichier CSV")
    uploaded = st.file_uploader("Importer un CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        result = predict(df, MODEL_PATH)
        output = pd.concat([df.reset_index(drop=True), result.reset_index(drop=True)], axis=1)
        st.dataframe(output, use_container_width=True)
        st.download_button(
            "Télécharger les prédictions",
            output.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )

else:
    st.subheader("À propos")
    st.write("FinAccess AI est un projet intégrateur de Science des Données. Le modèle estime la probabilité d'appartenance à une classe de possession de compte bancaire à partir de variables du profil fourni.")
    st.warning("Cette prédiction n'est pas une décision d'éligibilité bancaire ni une décision de crédit.")
