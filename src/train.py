"""Entraînement du modèle FoodPrice AI (Personne 2).

Fonctions exposées aux autres membres du groupe :
  - train_candidate(data, valid=None) -> {"model", "X_valid", "y_valid"}   [pour retrain.py]
  - evaluate_model(model, X, y)       -> dict MAE / RMSE / R2
  - save_model(model, metrics, ...)   -> écrit models/model_current.joblib + .json
  - python -m src.train               -> entraîne le modèle final et le sauvegarde
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import tempfile
import sys
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

# retrain.py ajoute seulement src/ au chemin Python : on ajoute aussi la racine du projet
# pour que les imports « from src.xxx » fonctionnent quel que soit le point de départ.
_ROOT_DIR = str(Path(__file__).resolve().parents[1])
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from src.models import MODEL_LABELS, build_model, load_config
from src.preprocessing import (
    RAW_TRAIN_PATH, ROOT, TEST_PATH, TRAIN_PATH, VALID_PATH, clean_data, ensure_splits, load_data, split_xy,
)
from src.reference import build_reference, save_reference
from src.schema import INPUT_COLUMNS, TARGET, validate_dataframe

MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "model_current.joblib"
META_PATH = MODELS_DIR / "model_current.json"
ARCHIVE_DIR = MODELS_DIR / "archive"
DEFAULT_VALID_PATH = VALID_PATH   # data/validation.parquet : jeu de validation figé


def _prepare(df: pd.DataFrame, name: str) -> pd.DataFrame:
    problems = validate_dataframe(df, with_target=True)
    if problems:
        raise ValueError(f"Données '{name}' invalides : " + " | ".join(problems))
    return clean_data(df)


def fit_candidate(data: pd.DataFrame, valid: pd.DataFrame | None = None,
                  model_name: str | None = None, params: dict | None = None):
    """Entraîne un modèle candidat.

    data  : DataFrame d'entraînement (train + éventuels nouveaux lots validés).
            Doit contenir les colonnes de src.schema.REQUIRED_COLUMNS.
    valid : jeu de validation FIGÉ. Par défaut data/validation.parquet (le test.parquet reste réservé à l'évaluation finale). Il ne doit jamais
            être utilisé pour entraîner, afin que l'ancien et le nouveau modèle
            soient comparés sur exactement les mêmes lignes.
    Retourne (pipeline, X_valid, y_valid). pipeline.predict(X_valid) renvoie des FCFA.
    """
    train_df = _prepare(data, "entraînement")
    if valid is None:
        ensure_splits()
        valid = load_data(DEFAULT_VALID_PATH)
    valid_df = _prepare(valid, "validation")

    config = load_config()
    name = model_name or config["model_name"]
    model_params = params if params is not None else config["params"]

    X_train, y_train = split_xy(train_df)
    X_valid, y_valid = split_xy(valid_df)
    model = build_model(name, model_params)
    model.fit(X_train, y_train)
    return model, X_valid, y_valid


def train_candidate(data: pd.DataFrame, valid: pd.DataFrame | None = None,
                    model_name: str | None = None, params: dict | None = None) -> dict:
    """Interface attendue par src/retrain.py : retourne un dictionnaire.

    {"model": pipeline, "X_valid": DataFrame, "y_valid": Series}
    (même entraînement que fit_candidate, seul le format de retour change).
    """
    model, X_valid, y_valid = fit_candidate(data, valid, model_name, params)
    return {"model": model, "X_valid": X_valid, "y_valid": y_valid}


def evaluate_model(model, X, y) -> dict:
    pred = model.predict(X)
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(root_mean_squared_error(y, pred)),
        "R2": float(r2_score(y, pred)),
    }


def _atomic_dump(obj, path: Path, as_json: bool = False):
    """Écrit dans un fichier temporaire puis renomme : jamais de fichier à moitié écrit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        if as_json:
            Path(tmp).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            joblib.dump(obj, tmp, compress=3)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def read_metadata(meta_path: Path = META_PATH) -> dict | None:
    return json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None


def save_model(model, metrics: dict, model_name: str, params: dict, n_train: int,
               models_dir: Path = MODELS_DIR, archive_previous: bool = True,
               metrics_test: dict | None = None) -> dict:
    """Sauvegarde le modèle actif + ses métadonnées ; archive l'ancienne version."""
    models_dir = Path(models_dir)
    model_path = models_dir / "model_current.joblib"
    meta_path = models_dir / "model_current.json"
    archive_dir = models_dir / "archive"

    previous = read_metadata(meta_path)
    if archive_previous and model_path.exists():
        archive_dir.mkdir(parents=True, exist_ok=True)
        stamp = (previous or {}).get("trained_at", datetime.now().isoformat(timespec="seconds"))
        stamp = stamp.replace(":", "").replace("-", "").replace("T", "_")
        shutil.copy2(model_path, archive_dir / f"model_{stamp}.joblib")
        if previous:
            (archive_dir / f"model_{stamp}.json").write_text(
                json.dumps(previous, indent=2, ensure_ascii=False), encoding="utf-8")

    meta = {
        "version": (previous or {}).get("version", 0) + 1,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name,
        "model_label": MODEL_LABELS.get(model_name, model_name),
        "params": params,
        "target": TARGET,
        "input_columns": INPUT_COLUMNS,
        "n_train_rows": n_train,
        "metrics_validation": metrics,
        "metrics_test": metrics_test,   # renseigné seulement à l'entraînement initial
        "libraries": {"scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__,
                      "pandas": pd.__version__, "joblib": joblib.__version__},
    }
    _atomic_dump(model, model_path)
    _atomic_dump(meta, meta_path, as_json=True)
    return meta


def main():
    argparse.ArgumentParser(description="Entraîne et sauvegarde le modèle final.").parse_args()
    ensure_splits()
    config = load_config()
    train = load_data(TRAIN_PATH)
    print(f"Entraînement : {MODEL_LABELS[config['model_name']]} | {config['params']}")
    model, X_valid, y_valid = fit_candidate(train)
    metrics = evaluate_model(model, X_valid, y_valid)
    print("Validation figée :", {k: round(v, 3) for k, v in metrics.items()})
    # Test final : évalué ici une seule fois, à titre de rapport (jamais pour décider)
    X_test, y_test = split_xy(clean_data(load_data(TEST_PATH)))
    metrics_test = evaluate_model(model, X_test, y_test)
    print("Test final       :", {k: round(v, 3) for k, v in metrics_test.items()})
    meta = save_model(model, metrics, config["model_name"], config["params"],
                      n_train=len(clean_data(train)), metrics_test=metrics_test)
    save_reference(build_reference(clean_data(load_data(RAW_TRAIN_PATH))))
    print(f"Modèle sauvegardé : {MODEL_PATH} (version {meta['version']})")
    print(f"Table de référence : {ROOT / 'reference.json'}")


if __name__ == "__main__":
    main()
