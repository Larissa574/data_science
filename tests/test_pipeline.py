"""Tests du pipeline ML (pytest). Lancer depuis la racine : python -m pytest tests"""
import json

import numpy as np
import pandas as pd
import pytest

from src.models import build_model
from src.preprocessing import (
    RAW_TRAIN_PATH, TEST_PATH, TRAIN_PATH, VALID_PATH, clean_data, ensure_splits, load_data, split_xy,
)
from src.reference import build_reference, make_input_row
from src.schema import INPUT_COLUMNS, REQUIRED_COLUMNS, validate_dataframe
from src.train import evaluate_model, fit_candidate, save_model, train_candidate

SMALL = {"n_estimators": 30, "learning_rate": 0.1, "num_leaves": 15}


@pytest.fixture(scope="module")
def sample():
    df = load_data(RAW_TRAIN_PATH)
    return df.sample(4000, random_state=0).reset_index(drop=True)


def test_validate_ok_and_errors(sample):
    assert validate_dataframe(sample) == []
    assert "Colonnes manquantes" in validate_dataframe(sample.drop(columns=["price"]))[0]
    bad = sample.copy()
    bad.loc[0, "price"] = -5
    assert any("Prix <= 0" in p for p in validate_dataframe(bad))


def test_train_candidate_contract(sample):
    train, valid = sample.iloc[:3000], sample.iloc[3000:]
    model, X_valid, y_valid = fit_candidate(train, valid, "lightgbm", SMALL)
    assert list(X_valid.columns) == INPUT_COLUMNS
    assert len(X_valid) == len(y_valid) == len(valid)
    metrics = evaluate_model(model, X_valid, y_valid)
    assert set(metrics) == {"MAE", "RMSE", "R2"}
    assert metrics["MAE"] < 500          # bien meilleur que la baseline (~425 FCFA sur tout le jeu)


def test_train_candidate_rejects_bad_data(sample):
    with pytest.raises(ValueError):
        train_candidate(sample.drop(columns=["market"]), sample)


@pytest.mark.parametrize("name,params", [
    ("ridge", {"alpha": 1.0}), ("random_forest", {"n_estimators": 10}), ("lightgbm", SMALL)])
def test_all_models_predict_positive_fcfa(sample, name, params):
    X, y = split_xy(clean_data(sample))
    model = build_model(name, params).fit(X, y)
    pred = model.predict(X.head(20))
    assert pred.shape == (20,) and np.isfinite(pred).all()


def test_unseen_market_does_not_crash(sample):
    X, y = split_xy(clean_data(sample))
    model = build_model("lightgbm", SMALL).fit(X, y)
    row = X.head(1).copy()
    row["market"] = "Marché jamais vu"
    assert np.isfinite(model.predict(row)).all()


def test_save_model_versions_and_archives(sample, tmp_path):
    X, y = split_xy(clean_data(sample))
    model = build_model("lightgbm", SMALL).fit(X, y)
    m1 = save_model(model, {"MAE": 1.0}, "lightgbm", SMALL, len(X), models_dir=tmp_path)
    m2 = save_model(model, {"MAE": 0.9}, "lightgbm", SMALL, len(X), models_dir=tmp_path)
    assert (m1["version"], m2["version"]) == (1, 2)
    assert (tmp_path / "model_current.joblib").exists()
    assert len(list((tmp_path / "archive").glob("*.joblib"))) == 1
    assert json.loads((tmp_path / "model_current.json").read_text())["version"] == 2


def test_reference_builds_valid_input_row(sample):
    ref = build_reference(clean_data(sample))
    market, commodity = next(iter(ref["markets"])), next(iter(ref["commodities"]))
    row = make_input_row(ref, market, commodity, "2026-03-15")
    assert list(row.columns) == INPUT_COLUMNS and len(row) == 1
    assert validate_dataframe(row.assign(price=1.0), with_target=True) == []
    assert set(REQUIRED_COLUMNS) >= set(INPUT_COLUMNS)


def test_splits_are_disjoint_and_same_schema():
    ensure_splits()
    fit, valid, test = (load_data(p) for p in (TRAIN_PATH, VALID_PATH, TEST_PATH))
    raw = clean_data(load_data(RAW_TRAIN_PATH))
    assert len(fit) + len(valid) == len(raw)
    assert list(fit.columns) == list(valid.columns) == list(test.columns)
    key = ["date", "market", "commodity", "unit", "pricetype"]
    assert len(fit[key].merge(valid[key])) == 0      # aucune ligne commune entre entraînement et validation
    assert validate_dataframe(valid) == [] and validate_dataframe(fit) == []


def test_train_candidate_matches_retrain_interface(sample):
    """retrain.py attend un dictionnaire {model, X_valid, y_valid}."""
    from src.retrain import validate_training_bundle
    bundle = train_candidate(sample.iloc[:3000], sample.iloc[3000:], "lightgbm", SMALL)
    model, X_valid, y_valid = validate_training_bundle(bundle)
    assert len(X_valid) == len(y_valid) == len(sample.iloc[3000:])
