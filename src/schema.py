"""Contrat de données FoodPrice AI (Personne 2 -> Personnes 3 et 4).

Ce fichier est la source unique de vérité sur les colonnes attendues.
- L'application (Personne 3) envoie un DataFrame avec INPUT_COLUMNS.
- retrain.py (Personne 4) valide les nouveaux lots avec validate_dataframe().
"""
import pandas as pd

TARGET = "price"

# Colonnes brutes attendues par le modèle (entrée de pipeline.predict)
CATEGORICAL_FEATURES = [
    "admin1", "admin2", "market", "category", "commodity", "unit", "pricetype",
]
NUMERIC_FEATURES = ["latitude", "longitude"]
DATE_COLUMN = "date"
INPUT_COLUMNS = [DATE_COLUMN] + CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Colonnes nécessaires dans un fichier de données d'entraînement (entrées + cible)
REQUIRED_COLUMNS = INPUT_COLUMNS + [TARGET]

# Colonnes volontairement exclues
#  - usdprice : corrélation 0.997 avec la cible (fuite de données)
#  - currency, priceflag, esa_source, esa_processed : constantes / sans intérêt
#  - market_id, commodity_id : redondants avec market / commodity
EXCLUDED_COLUMNS = [
    "usdprice", "currency", "priceflag", "esa_source", "esa_processed",
    "market_id", "commodity_id",
]


def validate_dataframe(df: pd.DataFrame, with_target: bool = True) -> list:
    """Retourne la liste des problèmes détectés (liste vide = données valides)."""
    problems = []
    required = REQUIRED_COLUMNS if with_target else INPUT_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        problems.append(f"Colonnes manquantes : {missing}")
        return problems  # inutile de continuer sans les colonnes

    if df.empty:
        problems.append("Le fichier est vide.")
        return problems

    if not pd.api.types.is_datetime64_any_dtype(df[DATE_COLUMN]):
        try:
            pd.to_datetime(df[DATE_COLUMN])
        except Exception:
            problems.append("La colonne 'date' n'est pas convertible en date.")

    for col in NUMERIC_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[col]):
            problems.append(f"La colonne '{col}' doit être numérique.")

    if with_target:
        if not pd.api.types.is_numeric_dtype(df[TARGET]):
            problems.append("La colonne 'price' doit être numérique.")
        else:
            if df[TARGET].isna().any():
                problems.append("Valeurs manquantes dans 'price'.")
            if (df[TARGET] <= 0).any():
                problems.append("Prix <= 0 détectés dans 'price'.")

    for col in INPUT_COLUMNS:
        n_na = int(df[col].isna().sum())
        if n_na:
            problems.append(f"{n_na} valeur(s) manquante(s) dans '{col}'.")
    return problems
