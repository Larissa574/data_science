"""Tests for the FoodPrice AI retraining orchestration."""

import pandas as pd

from src.retrain import candidate_is_better, load_and_validate_data


def test_validation_removes_invalid_rows_and_duplicates(tmp_path):
    frame = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-01", "2025-01-03", "not-a-date"],
            "commodity": ["Maïs", "Maïs", "Riz", "Riz"],
            "unit": ["kg", "kg", "kg", "kg"],
            "price": [300, 300, -20, 450],
        }
    )
    parquet_path = tmp_path / "prices.parquet"
    frame.to_parquet(parquet_path, index=False)

    cleaned, quality, paths = load_and_validate_data([str(parquet_path)])

    assert len(paths) == 1
    assert len(cleaned) == 1
    assert quality["rows_loaded"] == 4
    assert quality["invalid_rows_removed"] == 2
    assert quality["duplicates_removed"] == 1
    assert cleaned.iloc[0]["price"] == 300


def test_candidate_is_accepted_only_if_both_errors_do_not_increase():
    incumbent = {"mae": 40.0, "rmse": 60.0, "r2": 0.7}
    better = {"mae": 35.0, "rmse": 55.0, "r2": 0.75}
    worse_rmse = {"mae": 30.0, "rmse": 65.0, "r2": 0.72}
    identical = {"mae": 40.0, "rmse": 60.0, "r2": 0.7}

    assert candidate_is_better(better, incumbent)
    assert not candidate_is_better(worse_rmse, incumbent)
    assert not candidate_is_better(identical, incumbent)
