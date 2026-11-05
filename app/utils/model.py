import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import sys
import json

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
# The pickled pipeline references classes from the `src` package
# (e.g. src.preprocessing.DateFeatures): the repo root must be importable.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "models" / "model_current.joblib"

FEATURES = [
    "date", "admin1", "admin2", "market", "category",
    "commodity", "unit", "pricetype", "latitude", "longitude",
]
LOG_TARGET = False
# ------------------------------------

def load_metadata():
    """Return the model metadata dict, or None if the file is absent."""
    path = MODEL_PATH.with_suffix(".json")
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

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

def load_metadata():
    """Return the model metadata dict, or None if the file is absent."""
    path = MODEL_PATH.with_suffix(".json")
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))