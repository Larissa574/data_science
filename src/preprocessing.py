"""Preprocessing utilities for the FinAccess AI project."""

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic, non-destructive cleaning."""
    data = df.copy()
    data = data.drop_duplicates().reset_index(drop=True)
    data.columns = [str(col).strip() for col in data.columns]
    return data


def prepare_target(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features and target column."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")

    data = df.dropna(subset=[target]).copy()
    X = data.drop(columns=[target])
    y = data[target]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a preprocessing pipeline for numerical and categorical features."""
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
