"""Automatic data-change detection, drift monitoring and controlled retraining.

Default workflow:
- train from data/train_fit.parquet (the fixed validation and test sets stay separate);
- add new Parquet batches to data/incoming/;
- run once, or keep --watch active to poll for new observations;
- train only when the cleaned dataset actually changes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import re
import shutil
import sys
import time
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

DEFAULT_BASE_DATA = ROOT / "data" / "train_fit.parquet"
DEFAULT_INCOMING_DIR = ROOT / "data" / "incoming"
DEFAULT_MODEL_PATH = ROOT / "models" / "model_current.joblib"
DEFAULT_LOG_PATH = ROOT / "logs" / "retraining_history.csv"
DEFAULT_STATE_PATH = ROOT / "logs" / "retraining_state.json"
DEFAULT_DRIFT_REFERENCE_PATH = ROOT / "logs" / "data_drift_reference.json"

REQUIRED_COLUMNS = {"date", "commodity", "unit", "price"}
MAX_CATEGORIES = 50
PSI_STABLE = 0.10
PSI_SIGNIFICANT = 0.20

HISTORY_FIELDS = [
    "run_at_utc", "data_files", "data_fingerprint", "rows_loaded", "rows_valid",
    "invalid_rows_removed", "duplicates_removed", "drift_max_psi", "drift_status",
    "drift_details", "candidate_mae", "candidate_rmse", "candidate_r2",
    "current_mae", "current_rmse", "current_r2", "decision", "reason",
]


def _is_reserved_evaluation_file(path: Path) -> bool:
    """Never ingest validation/test shards into the training data."""
    stem = path.stem.lower()
    return re.search(r"(^|[-_.])(test|valid|validation)([-_.]|$)", stem) is not None


def expand_data_paths(values: list[str]) -> list[Path]:
    """Expand files/directories/wildcards, excluding reserved evaluation files."""
    import glob

    found: list[Path] = []
    raw_dir = (ROOT / "data" / "raw").resolve()

    for value in values:
        candidate = Path(value)
        if candidate.is_dir():
            if candidate.resolve() == raw_dir:
                raise ValueError(
                    "Ne passe pas data/raw au réentraînement : ce dossier contient "
                    "le jeu brut complet. Utilise data/train_fit.parquet, qui exclut "
                    "la validation, puis ajoute les nouveaux lots dans data/incoming/."
                )
            found.extend(candidate.glob("*.parquet"))
            found.extend(candidate.glob("*.pq"))
        elif any(char in value for char in "*?["):
            found.extend(Path(item) for item in glob.glob(value, recursive=True))
        elif candidate.is_file():
            found.append(candidate)

    unique: dict[str, Path] = {}
    for path in found:
        if not path.is_file() or path.suffix.lower() not in {".parquet", ".pq"}:
            continue
        if _is_reserved_evaluation_file(path):
            continue
        unique[str(path.resolve())] = path.resolve()

    return sorted(unique.values())


def load_and_validate_data(values: list[str]) -> tuple[pd.DataFrame, dict[str, int], list[Path]]:
    """Load Parquet files, validate essential fields and remove unusable rows."""
    paths = expand_data_paths(values)
    if not paths:
        raise FileNotFoundError(
            "Aucun fichier Parquet d'entraînement trouvé. Utilise data/train_fit.parquet "
            "et place les nouveaux lots dans data/incoming/."
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


def dataframe_fingerprint(data: pd.DataFrame) -> str:
    """Hash cleaned rows independently of file names and row order."""
    frame = data.reindex(sorted(data.columns), axis=1)
    row_hashes = pd.util.hash_pandas_object(
        frame, index=False, categorize=True
    ).to_numpy(dtype=np.uint64)
    row_hashes.sort()

    digest = hashlib.sha256()
    digest.update("\n".join(map(str, frame.columns)).encode("utf-8"))
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def _numeric_values(series: pd.Series, column: str) -> np.ndarray:
    if column == "date":
        values = pd.to_datetime(series, errors="coerce")
        numeric = values.astype("int64").to_numpy(dtype=np.float64) / 86_400_000_000_000
        numeric[values.isna().to_numpy()] = np.nan
    else:
        numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    return numeric[np.isfinite(numeric)]


def _categorical_values(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("__MISSING__").astype(str)


def build_drift_reference(data: pd.DataFrame) -> dict[str, Any]:
    """Build a fixed baseline profile from the original training partition."""
    from src.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES

    features: dict[str, Any] = {}

    for column in [*CATEGORICAL_FEATURES]:
        if column not in data.columns:
            continue
        values = _categorical_values(data[column])
        counts = values.value_counts(normalize=True)
        categories = counts.head(MAX_CATEGORIES).index.tolist()
        expected = {category: float(counts.get(category, 0.0)) for category in categories}
        expected["__OTHER__"] = float(max(0.0, 1.0 - sum(expected.values())))
        features[column] = {"kind": "categorical", "expected": expected}

    for column in ["date", *NUMERIC_FEATURES, "price"]:
        if column not in data.columns:
            continue
        values = _numeric_values(data[column], column)
        if len(values) == 0:
            continue
        quantiles = np.quantile(values, np.linspace(0.0, 1.0, 11))
        cut_points = np.unique(quantiles[1:-1]).astype(float).tolist()
        bin_ids = np.searchsorted(np.asarray(cut_points), values, side="right")
        counts = np.bincount(bin_ids, minlength=len(cut_points) + 1).astype(float)
        expected = (counts / counts.sum()).tolist()
        features[column] = {
            "kind": "numeric",
            "cut_points": cut_points,
            "expected": expected,
        }

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(data)),
        "features": features,
    }


def _population_stability_index(expected: list[float], actual: list[float]) -> float:
    eps = 1e-6
    expected_array = np.asarray(expected, dtype=float)
    actual_array = np.asarray(actual, dtype=float)
    expected_array = np.clip(expected_array, eps, None)
    actual_array = np.clip(actual_array, eps, None)
    expected_array /= expected_array.sum()
    actual_array /= actual_array.sum()
    return float(np.sum((actual_array - expected_array) * np.log(actual_array / expected_array)))


def calculate_data_drift(data: pd.DataFrame, reference: dict[str, Any]) -> dict[str, Any]:
    """Compare the incoming batch with the fixed baseline using PSI per feature."""
    results: dict[str, float] = {}

    for column, profile in reference.get("features", {}).items():
        if column not in data.columns:
            continue

        if profile["kind"] == "categorical":
            known = profile["expected"]
            categories = [key for key in known if key != "__OTHER__"]
            values = _categorical_values(data[column])
            mapped = values.where(values.isin(categories), "__OTHER__")
            actual_counts = mapped.value_counts(normalize=True)
            keys = [*categories, "__OTHER__"]
            expected = [float(known.get(key, 0.0)) for key in keys]
            actual = [float(actual_counts.get(key, 0.0)) for key in keys]
        else:
            values = _numeric_values(data[column], column)
            if len(values) == 0:
                continue
            cut_points = np.asarray(profile["cut_points"], dtype=float)
            bin_ids = np.searchsorted(cut_points, values, side="right")
            counts = np.bincount(bin_ids, minlength=len(cut_points) + 1).astype(float)
            actual = (counts / counts.sum()).tolist()
            expected = [float(value) for value in profile["expected"]]

        results[column] = _population_stability_index(expected, actual)

    if not results:
        return {"status": "unavailable", "max_psi": None, "features": {}}

    max_psi = max(results.values())
    if max_psi < PSI_STABLE:
        status = "stable"
    elif max_psi < PSI_SIGNIFICANT:
        status = "moderate"
    else:
        status = "significant"

    return {
        "status": status,
        "max_psi": float(max_psi),
        "features": {key: round(value, 6) for key, value in sorted(results.items())},
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


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
    """Accept only when MAE and RMSE both improve or stay equal, with one strict gain."""
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


def update_model_metadata(model_path: Path, metrics: dict[str, float | None], n_train: int) -> None:
    """Keep the sidecar JSON synchronized when a candidate becomes active."""
    metadata_path = model_path.with_suffix(".json")
    metadata = _read_json(metadata_path) or {}
    metadata["version"] = int(metadata.get("version", 0)) + 1
    metadata["trained_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metadata["n_train_rows"] = int(n_train)
    metadata["metrics_validation"] = {
        "MAE": metrics["mae"],
        "RMSE": metrics["rmse"],
        "R2": metrics["r2"],
    }
    # The final test set is deliberately not re-used during retraining.
    metadata["metrics_test"] = None
    metadata["last_retraining"] = "automatic"
    _write_json_atomic(metadata_path, metadata)


def append_history(log_path: Path, record: dict[str, Any]) -> None:
    """Append one completed attempt to the local CSV history."""
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


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def run_retraining(args: argparse.Namespace) -> int:
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record = empty_record(started_at, args.data)
    should_log = True

    try:
        # Generate the fixed train/validation split if this is the default training file.
        if any(_resolve_path(value) == DEFAULT_BASE_DATA for value in args.data):
            from src.preprocessing import ensure_splits
            ensure_splits()

        incoming_dir = _resolve_path(args.incoming_dir)
        incoming_dir.mkdir(parents=True, exist_ok=True)

        incoming_paths = expand_data_paths([str(incoming_dir)])
        all_values = [*args.data, str(incoming_dir)]
        data, quality, paths = load_and_validate_data(all_values)
        record.update(quality)
        record["data_files"] = ";".join(str(path) for path in paths)

        fingerprint = dataframe_fingerprint(data)
        record["data_fingerprint"] = fingerprint

        state_path = _resolve_path(args.state_path)
        state = _read_json(state_path)
        if state and state.get("data_fingerprint") == fingerprint:
            print("Aucune nouvelle observation détectée : réentraînement ignoré.")
            should_log = False
            return 0

        # Drift is measured on incoming batches only, not on the historical training set.
        if incoming_paths:
            drift_data, _, _ = load_and_validate_data([str(path) for path in incoming_paths])
        else:
            base_data, _, _ = load_and_validate_data(args.data)
            drift_data = base_data

        reference_path = _resolve_path(args.drift_reference_path)
        drift_reference = _read_json(reference_path)
        if drift_reference is None:
            base_data, _, _ = load_and_validate_data(args.data)
            drift_reference = build_drift_reference(base_data)

        drift = calculate_data_drift(drift_data, drift_reference)
        record.update({
            "drift_max_psi": drift["max_psi"],
            "drift_status": drift["status"],
            "drift_details": json.dumps(drift["features"], ensure_ascii=False),
        })

        trainer = load_trainer(args.trainer)
        bundle = trainer(data)
        candidate_model, X_valid, y_valid = validate_training_bundle(bundle)
        candidate_metrics = regression_metrics(candidate_model, X_valid, y_valid)

        record.update({
            "candidate_mae": candidate_metrics["mae"],
            "candidate_rmse": candidate_metrics["rmse"],
            "candidate_r2": candidate_metrics["r2"],
        })

        model_path = _resolve_path(args.model_path)
        promoted = False
        if not model_path.exists():
            archive = promote_model(candidate_model, model_path)
            promoted = True
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
                promoted = True
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

        if promoted:
            update_model_metadata(model_path, candidate_metrics, len(data))

        # Save the fixed baseline only after a successful training/evaluation cycle.
        if not reference_path.exists():
            _write_json_atomic(reference_path, drift_reference)

        _write_json_atomic(state_path, {
            "data_fingerprint": fingerprint,
            "last_processed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rows": int(len(data)),
            "data_files": [str(path) for path in paths],
            "drift": drift,
            "decision": record["decision"],
        })

        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 0

    except Exception as error:
        record["reason"] = f"{type(error).__name__}: {error}"
        print(f"ÉCHEC DU RÉENTRAÎNEMENT : {record['reason']}", file=sys.stderr)
        return 1

    finally:
        if should_log:
            append_history(_resolve_path(args.log_path), record)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Surveille les données FoodPrice AI et lance un réentraînement contrôlé."
    )
    parser.add_argument(
        "--data",
        nargs="+",
        default=[str(DEFAULT_BASE_DATA)],
        help="Données historiques d'entraînement (par défaut : data/train_fit.parquet).",
    )
    parser.add_argument(
        "--incoming-dir",
        default=str(DEFAULT_INCOMING_DIR),
        help="Dossier surveillé contenant les nouveaux lots Parquet (défaut : data/incoming).",
    )
    parser.add_argument(
        "--trainer",
        default="train:train_candidate",
        help="Fonction d'entraînement au format module:fonction.",
    )
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--log-path", default=str(DEFAULT_LOG_PATH))
    parser.add_argument("--state-path", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--drift-reference-path", default=str(DEFAULT_DRIFT_REFERENCE_PATH))
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Surveille en continu le dossier d'arrivée et vérifie les changements.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=30,
        help="Fréquence de vérification en mode --watch (défaut : 30 secondes).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.watch:
        return run_retraining(args)

    if args.poll_seconds < 5:
        raise ValueError("--poll-seconds doit être supérieur ou égal à 5.")

    print(
        f"Surveillance active : {args.incoming_dir} "
        f"(vérification toutes les {args.poll_seconds} secondes)."
    )
    try:
        while True:
            run_retraining(args)
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("\nSurveillance arrêtée par l'utilisateur.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
