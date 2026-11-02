"""Prediction utilities for the FinAccess AI application."""

from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/model.joblib")


def load_model(path: str | Path = MODEL_PATH):
    """Load the trained model pipeline."""
    return joblib.load(path)


def predict(data: dict | pd.DataFrame, model_path: str | Path = MODEL_PATH):
    """Predict bank-account ownership for one or more observations."""
    model = load_model(model_path)
    X = pd.DataFrame([data]) if isinstance(data, dict) else data.copy()

    predictions = model.predict(X)
    result = pd.DataFrame({"prediction": predictions})

    if hasattr(model, "predict_proba"):
        result["probability"] = model.predict_proba(X).max(axis=1)

    return result


if __name__ == "__main__":
    print("Prediction module ready. Import predict() from the application.")
