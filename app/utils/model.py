from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "artifacts" / "model.joblib"

# --- A ALIGNER AVEC LA PERSONNE 2 ---
FEATURES = [
    "admin1", "admin2", "market", "latitude", "longitude",
    "category", "commodity", "unit", "pricetype", "year", "month",
]
LOG_TARGET = False
# ------------------------------------


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Return the trained pipeline, or None if the file does not exist yet."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create year and month from a 'date' column (reused by the batch page)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df


def predict(model, X: pd.DataFrame) -> np.ndarray:
    """Predict prices in XOF from a DataFrame containing FEATURES."""
    pred = model.predict(X[FEATURES])
    if LOG_TARGET:
        pred = np.expm1(pred)
    return np.clip(pred, 0, None)