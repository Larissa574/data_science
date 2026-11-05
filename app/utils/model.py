from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "model_current.joblib"

FEATURES = [
    "date", "admin1", "admin2", "market", "latitude", "longitude",
    "category", "commodity", "unit", "pricetype",
]
LOG_TARGET = False
# ------------------------------------


@st.cache_resource(show_spinner="Loading model...")
def _load(path: str, mtime: float):
    """Cached loader: a new modification time forces a reload of the model."""
    return joblib.load(path)


def load_model():
    """Return the trained pipeline, or None if the file does not exist yet."""
    if not MODEL_PATH.exists():
        return None
    return _load(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)


def predict(model, X: pd.DataFrame) -> np.ndarray:
    """Predict prices in XOF from a DataFrame containing FEATURES."""
    pred = model.predict(X[FEATURES])
    if LOG_TARGET:
        pred = np.expm1(pred)
    return np.clip(pred, 0, None)