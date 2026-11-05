"""Controlled retraining for FoodPrice AI.

Expected interface from src/train.py:
    def train_candidate(data: pandas.DataFrame) -> dict:
        return {
            "model": fitted_sklearn_pipeline,
            "X_valid": validation_features,
            "y_valid": validation_target,
        }

The validation split must be temporal and must be the same split used to
evaluate both the candidate and the currently deployed model.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_DATA_DIR = ROOT / "data" / "raw"
DEFAULT_MODEL_PATH = ROOT / "models" / "model_current.joblib"
DEFAULT_LOG_PATH = ROOT / "logs" / "retraining_history.csv"

# These fields are necessary to identify a dated food-price observation.
REQUIRED_COLUMNS = {"date", "commodity", "unit", "price"}
HISTORY_FIELDS = [
    "run_at_utc",
    "data_files",
    "rows_loaded",
    "rows_valid",
    "invalid_rows_removed",
    "duplicates_removed",
    "candidate_mae",
    "candidate_rmse",
    "candidate_r2",
    "current_mae",
    "current_rmse",
    "current_r2",
    "decision",
    "reason",
]


def expand_data_paths(values: list[str]) -> list[Path]:
    """Expand files, directories and shell-style wildcard patterns."""
    import glob

    found: list[Path] = []
    for value in values:
        candidate = Path(value)
        if candidate.is_dir():
            found.extend(candidate.glob("*.parquet"))
            found.extend(candidate.glob("*.pq"))
        elif any(char in value for char in "*?["):
            found.extend(Path(item) for item in glob.glob(value, recursive=True))
        elif candidate.is_file():
            found.append(candidate)

    unique: dict[str, Path] = {}
    for path in found:
        if path.is_file() and path.suffix.lower() in {".parquet", ".pq"}:
            unique[str(path.resolve())] = path.resolve()

    return sorted(unique.values())


def load_and_validate_data(values: list[str]) -> tuple[pd.DataFrame, dict[str, int], list[Path]]:
    """Load Parquet files, validate essential fields and remove unusable rows."""
    paths = expand_data_paths(values)
    if not paths:
        raise FileNotFoundError(
            "Aucun fichier Parquet trouvé. Place les fichiers dans data/raw/ "
            "ou indique leur chemin avec --data."
        )

    frames = [pd.read_parquet(path) for path in paths]
    data = pd.concat(frames, ignore_index=True)

    data.columns = [str(column).strip().lower() for column in data.columns]
    if data.columns.duplicated().any():
        duplicated = data.columns[data.columns.duplicated()].tolist()
        raise ValueError(f"Colonnes dupliquées après normalisation : {duplicated}")

    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(
            "Colonnes obligatoires absentes : "
            + ", ".join(sorted(missing))
            + ". Colonnes reçues : "
            + ", ".join(data.columns)
        )

    rows_loaded = len(data)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data["commodity"] = data["commodity"].astype("string").str.strip()
    data["unit"] = data["unit"].astype("string").str.strip()

    valid_mask = (
        data["date"].notna()
        & data["price"].notna()
        & np.isfinite(data["price"])
        & (data["price"] > 0)
        & data["commodity"].notna()
        & data["commodity"].ne("")
        & data["unit"].notna()
        & data["unit"].ne("")
    )

    invalid_rows_removed = int((~valid_mask).sum())
    data = data.loc[valid_mask].copy()

    before_deduplication = len(data)
    data = data.drop_duplicates().reset_index(drop=True)
    duplicates_removed = before_deduplication - len(data)

    if data.empty:
        raise ValueError("Aucune observation exploitable après validation.")

    quality = {
        "rows_loaded": rows_loaded,
        "rows_valid": len(data),
        "invalid_rows_removed": invalid_rows_removed,
        "duplicates_removed": duplicates_removed,
    }
    return data, quality, paths


def load_trainer(specification: str):
    """Load the team's training function using 'module:function' notation."""
    if ":" not in specification:
        raise ValueError("Le trainer doit être indiqué sous la forme module:fonction.")

    module_name, function_name = specification.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    trainer = getattr(module, function_name, None)

    if not callable(trainer):
        raise RuntimeError(
            f"La fonction {specification} est introuvable. "
            "La personne 2 doit exposer train_candidate(dataframe) dans src/train.py."
        )
    return trainer


def validate_training_bundle(bundle: Any) -> tuple[Any, pd.DataFrame, pd.Series]:
    """Check the interface returned by the ML training module."""
    if not isinstance(bundle, dict):
        raise TypeError("train_candidate() doit retourner un dictionnaire.")

    missing = {"model", "X_valid", "y_valid"}.difference(bundle)
    if missing:
        raise ValueError(
            "Résultat de train_candidate() incomplet. Clés manquantes : "
            + ", ".join(sorted(missing))
        )

    model = bundle["model"]
    X_valid = bundle["X_valid"]
    y_valid = bundle["y_valid"]

    if not hasattr(model, "predict"):
        raise TypeError("Le modèle candidat doit fournir une méthode predict().")
    if not isinstance(X_valid, pd.DataFrame):
        raise TypeError("X_valid doit être un DataFrame pandas.")
    if len(X_valid) == 0 or len(y_valid) == 0 or len(X_valid) != len(y_valid):
        raise ValueError("Le jeu de validation est vide ou ses dimensions ne concordent pas.")

    return model, X_valid, pd.Series(y_valid).reset_index(drop=True)


def regression_metrics(model: Any, X_valid: pd.DataFrame, y_valid: pd.Series) -> dict[str, float | None]:
    """Evaluate a price-regression model on a shared validation set."""
    predictions = np.asarray(model.predict(X_valid), dtype=float).reshape(-1)
    actual = np.asarray(y_valid, dtype=float).reshape(-1)

    if len(predictions) != len(actual):
        raise ValueError("Le nombre de prédictions ne correspond pas au nombre de valeurs réelles.")
    if not np.isfinite(predictions).all() or not np.isfinite(actual).all():
        raise ValueError("Les prédictions ou les valeurs réelles contiennent NaN/inf.")

    mae = float(mean_absolute_error(actual, predictions))
    rmse = float(np.sqrt(mean_squared_error(actual, predictions)))
    r2 = float(r2_score(actual, predictions)) if len(actual) >= 2 else None

    return {"mae": mae, "rmse": rmse, "r2": r2}


def candidate_is_better(
    candidate: dict[str, float | None],
    incumbent: dict[str, float | None],
) -> bool:
    """Accept only when MAE and RMSE both improve (or stay equal), with one strict gain."""
    mae_ok = candidate["mae"] <= incumbent["mae"]
    rmse_ok = candidate["rmse"] <= incumbent["rmse"]
    strict_gain = (
        candidate["mae"] < incumbent["mae"]
        or candidate["rmse"] < incumbent["rmse"]
    )
    return bool(mae_ok and rmse_ok and strict_gain)


def promote_model(model: Any, model_path: Path) -> Path | None:
    """Write a candidate atomically and archive the previous active model."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path = None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    temporary_path = model_path.with_name(f".{model_path.stem}.{timestamp}.tmp")
    try:
        joblib.dump(model, temporary_path)

        if model_path.exists():
            archive_dir = model_path.parent / "archive"
            archive_dir.mkdir(parents=True, exist_ok=True)
            archive_path = archive_dir / f"{model_path.stem}_{timestamp}{model_path.suffix}"
            shutil.copy2(model_path, archive_path)

        os.replace(temporary_path, model_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return archive_path


def append_history(log_path: Path, record: dict[str, Any]) -> None:
    """Append one run to the local CSV history."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists() or log_path.stat().st_size == 0

    with log_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: record.get(field, "") for field in HISTORY_FIELDS})


def empty_record(started_at: str, data_values: list[str]) -> dict[str, Any]:
    return {
        "run_at_utc": started_at,
        "data_files": ";".join(data_values),
        "decision": "error",
        "reason": "",
    }


def run_retraining(args: argparse.Namespace) -> int:
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record = empty_record(started_at, args.data)

    try:
        data, quality, paths = load_and_validate_data(args.data)
        record.update(quality)
        record["data_files"] = ";".join(str(path) for path in paths)

        trainer = load_trainer(args.trainer)
        bundle = trainer(data)
        candidate_model, X_valid, y_valid = validate_training_bundle(bundle)
        candidate_metrics = regression_metrics(candidate_model, X_valid, y_valid)

        record.update({
            "candidate_mae": candidate_metrics["mae"],
            "candidate_rmse": candidate_metrics["rmse"],
            "candidate_r2": candidate_metrics["r2"],
        })

        model_path = Path(args.model_path)
        if not model_path.is_absolute():
            model_path = ROOT / model_path

        if not model_path.exists():
            archive = promote_model(candidate_model, model_path)
            record.update({
                "decision": "accepted_initial_model",
                "reason": "Aucun modèle actif compatible : premier modèle enregistré.",
            })
            if archive:
                record["reason"] += f" Ancien modèle archivé : {archive.name}."
        else:
            incumbent_model = joblib.load(model_path)
            incumbent_metrics = regression_metrics(incumbent_model, X_valid, y_valid)
            record.update({
                "current_mae": incumbent_metrics["mae"],
                "current_rmse": incumbent_metrics["rmse"],
                "current_r2": incumbent_metrics["r2"],
            })

            if candidate_is_better(candidate_metrics, incumbent_metrics):
                archive = promote_model(candidate_model, model_path)
                record.update({
                    "decision": "accepted",
                    "reason": "MAE et RMSE ne se dégradent pas et au moins une métrique s'améliore.",
                })
                if archive:
                    record["reason"] += f" Ancien modèle archivé : {archive.name}."
            else:
                record.update({
                    "decision": "rejected",
                    "reason": "Le candidat n'améliore pas simultanément MAE et RMSE. Modèle actif conservé.",
                })

        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 0

    except Exception as error:
        record["reason"] = f"{type(error).__name__}: {error}"
        print(f"ÉCHEC DU RÉENTRAÎNEMENT : {record['reason']}", file=sys.stderr)
        return 1

    finally:
        log_path = Path(args.log_path)
        if not log_path.is_absolute():
            log_path = ROOT / log_path
        append_history(log_path, record)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Valide les données et orchestre le réentraînement contrôlé de FoodPrice AI."
    )
    parser.add_argument(
        "--data",
        nargs="+",
        default=[str(DEFAULT_DATA_DIR)],
        help="Fichier(s) Parquet ou dossier contenant les Parquet. Fournir l'historique complet et les nouveaux lots.",
    )
    parser.add_argument(
        "--trainer",
        default="train:train_candidate",
        help="Fonction d'entraînement au format module:fonction (défaut : train:train_candidate).",
    )
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--log-path", default=str(DEFAULT_LOG_PATH))
    return parser


def main(argv: list[str] | None = None) -> int:
    return run_retraining(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
