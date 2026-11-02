"""Train and compare baseline models for FinAccess AI."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from preprocessing import build_preprocessor, clean_data, load_data, prepare_target


DATA_PATH = Path("data/financial_inclusion.csv")
MODEL_PATH = Path("models/model.joblib")
TARGET = "bank_account"


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    scores = {
        "precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
        "recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
        "f1": f1_score(y_test, predictions, average="weighted", zero_division=0),
    }

    if hasattr(model, "predict_proba") and y_test.nunique() == 2:
        probabilities = model.predict_proba(X_test)[:, 1]
        try:
            scores["roc_auc"] = roc_auc_score(y_test, probabilities)
        except ValueError:
            scores["roc_auc"] = None

    return scores


def main():
    df = clean_data(load_data(DATA_PATH))
    X, y = prepare_target(df, TARGET)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        ),
    }

    results = {}
    trained = {}

    for name, estimator in models.items():
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", estimator)]
        )
        pipeline.fit(X_train, y_train)
        results[name] = evaluate_model(pipeline, X_test, y_test)
        trained[name] = pipeline
        print(name, results[name])

    best_name = max(results, key=lambda name: results[name]["f1"])
    best_model = trained[best_name]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"Best model: {best_name}")
    print(f"Saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
