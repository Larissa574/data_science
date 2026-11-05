"""Définition des modèles candidats (Personne 2).

Un seul endroit où les modèles sont construits : compare_models.py (comparaison)
et train.py (entraînement final / réentraînement) utilisent build_model().
"""
from __future__ import annotations
import json
from pathlib import Path

from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from src.preprocessing import make_pipeline

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "src" / "model_config.json"
RANDOM_STATE = 42

MODEL_LABELS = {
    "baseline": "Baseline (médiane)",
    "ridge": "Ridge",
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM (gradient boosting)",
}

# Configuration utilisée si src/model_config.json n'existe pas encore
DEFAULT_CONFIG = {
    "model_name": "lightgbm",
    "params": {
        "n_estimators": 1000, "learning_rate": 0.05, "num_leaves": 127,
        "min_child_samples": 5, "subsample": 0.8, "subsample_freq": 1,
        "colsample_bytree": 0.8,
    },
}


def build_model(name: str, params: dict | None = None):
    """Retourne un pipeline scikit-learn NON entraîné (prétraitement + modèle).

    Toutes les sorties de predict() sont des prix en FCFA.
    """
    params = dict(params or {})
    if name == "baseline":
        return DummyRegressor(strategy="median")
    if name == "ridge":
        log_target = params.pop("log_target", False)
        return make_pipeline(Ridge(**params), scale_numeric=True, log_target=log_target)
    if name == "random_forest":
        return make_pipeline(
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params),
            encoding="ordinal",
        )
    if name == "lightgbm":
        return make_pipeline(
            LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, **params),
            encoding="onehot",
        )
    raise ValueError(f"Modèle inconnu : {name}")


def load_config() -> dict:
    """Modèle final retenu (écrit par compare_models.py), sinon valeur par défaut."""
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return DEFAULT_CONFIG
