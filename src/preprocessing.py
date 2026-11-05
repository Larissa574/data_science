"""Préparation des données et construction du prétraitement (Personne 2).

Tout le prétraitement vit DANS le pipeline scikit-learn : l'application et
retrain.py n'ont qu'à fournir des données brutes (INPUT_COLUMNS).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.schema import (
    CATEGORICAL_FEATURES, DATE_COLUMN, INPUT_COLUMNS, NUMERIC_FEATURES, TARGET,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# Trois jeux distincts :
#  - train_fit.parquet  : entraînement (80 % du train d'origine)
#  - validation.parquet : validation FIGÉE (20 % du train d'origine), utilisée pour
#                         comparer les modèles et pour accepter/refuser un réentraînement
#  - test.parquet       : test final, réservé à l'évaluation finale (jamais utilisé pour décider)
def _first_existing(*candidates: Path) -> Path:
    """Premier fichier existant (dépôt : data/raw/ ; copie locale : data/). Sinon le premier chemin."""
    return next((path for path in candidates if path.exists()), candidates[0])


# Les fichiers bruts du dépôt sont dans data/raw/ (noms Hugging Face) ; data/train.parquet reste accepté.
RAW_TRAIN_PATH = _first_existing(DATA_DIR / "raw" / "train-00000-of-00001.parquet", DATA_DIR / "train.parquet")
TRAIN_PATH = DATA_DIR / "train_fit.parquet"
VALID_PATH = DATA_DIR / "validation.parquet"
TEST_PATH = _first_existing(DATA_DIR / "raw" / "test-00000-of-00001.parquet", DATA_DIR / "test.parquet")
VALID_FRACTION = 0.2
SPLIT_SEED = 42


# ----------------------------------------------------------------------
# 1. Chargement et nettoyage
# ----------------------------------------------------------------------
def load_data(path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage des lignes inutilisables pour l'entraînement.

    - date convertie en datetime
    - suppression des doublons exacts
    - suppression des lignes sans prix ou avec prix <= 0
    - normalisation des espaces dans les colonnes texte
    Les valeurs extrêmes sont CONSERVÉES : ce sont de vrais prix (produits chers).
    """
    df = df.copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).str.strip()
    df = df.drop_duplicates()
    df = df[df[TARGET].notna() & (df[TARGET] > 0)]
    return df.reset_index(drop=True)


def ensure_splits(force: bool = False) -> None:
    """Crée train_fit.parquet et validation.parquet à partir de train.parquet.

    Découpage aléatoire reproductible (seed 42) fait UNE seule fois. Une fois créé,
    validation.parquet ne doit plus jamais changer ni être mélangé aux données d'entraînement.
    Idéalement : le fichier est créé par P2 puis partagé tel quel au groupe.
    """
    if TRAIN_PATH.exists() and VALID_PATH.exists() and not force:
        return
    from sklearn.model_selection import train_test_split
    df = clean_data(pd.read_parquet(RAW_TRAIN_PATH))
    fit, valid = train_test_split(df, test_size=VALID_FRACTION, random_state=SPLIT_SEED)
    fit.reset_index(drop=True).to_parquet(TRAIN_PATH, index=False)
    valid.reset_index(drop=True).to_parquet(VALID_PATH, index=False)


def split_xy(df: pd.DataFrame):
    return df[INPUT_COLUMNS].copy(), df[TARGET].astype(float).copy()


# ----------------------------------------------------------------------
# 2. Feature engineering
# ----------------------------------------------------------------------
class DateFeatures(BaseEstimator, TransformerMixin):
    """Transforme la colonne 'date' en variables numériques.

    year, month, month_sin/month_cos (saisonnalité cyclique) et
    t = année décimale (tendance / inflation dans le temps).
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        d = pd.to_datetime(X[DATE_COLUMN])
        month = d.dt.month
        out = pd.DataFrame(index=X.index)
        out["year"] = d.dt.year
        out["month"] = month
        out["month_sin"] = np.sin(2 * np.pi * month / 12)
        out["month_cos"] = np.cos(2 * np.pi * month / 12)
        out["t"] = d.dt.year + (month - 1) / 12
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(["year", "month", "month_sin", "month_cos", "t"])


def build_preprocessor(scale_numeric: bool = False,
                       encoding: str = "onehot") -> ColumnTransformer:
    """ColumnTransformer unique : dates + catégories + numériques.

    scale_numeric : True pour les modèles linéaires (Ridge).
    encoding      : "onehot" (Ridge, LightGBM) ou "ordinal" (arbres, plus rapide).
    Les catégories jamais vues ne font pas planter (inconnues ignorées / NaN).
    """
    if scale_numeric:
        date_tr = Pipeline([("dates", DateFeatures()), ("scaler", StandardScaler())])
        num_tr = StandardScaler()
    else:
        date_tr = DateFeatures()
        num_tr = "passthrough"

    if encoding == "onehot":
        make_enc = lambda: OneHotEncoder(handle_unknown="ignore")  # noqa: E731
    elif encoding == "ordinal":
        make_enc = lambda: OrdinalEncoder(  # noqa: E731
            handle_unknown="use_encoded_value", unknown_value=np.nan)
    else:
        raise ValueError(f"encoding inconnu : {encoding}")

    return ColumnTransformer(
        transformers=[
            ("date", date_tr, [DATE_COLUMN]),
            ("cat", make_enc(), CATEGORICAL_FEATURES),
            ("num", num_tr, NUMERIC_FEATURES),
        ],
        remainder="drop",
    )


def make_pipeline(regressor, scale_numeric: bool = False, log_target: bool = False,
                  encoding: str = "onehot"):
    """Pipeline complet : prétraitement + modèle. Prédit directement en FCFA.

    log_target=True : le modèle apprend log(1+prix) puis la sortie est reconvertie
    en FCFA (TransformedTargetRegressor).
    """
    pipe = Pipeline([
        ("prep", build_preprocessor(scale_numeric, encoding)),
        ("model", regressor),
    ])
    if log_target:
        return TransformedTargetRegressor(
            regressor=pipe, func=np.log1p, inverse_func=np.expm1, check_inverse=False,
        )
    return pipe
