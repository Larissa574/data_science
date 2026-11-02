"""Simple automated retraining script for FinAccess AI.

Usage:
    python src/retrain.py

The script retrains the project model from the current dataset and replaces the
saved model when training succeeds. Scheduling is intentionally external so the
same script can be run with Windows Task Scheduler, cron, or another scheduler.
"""

from pathlib import Path
import sys

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import DATA_PATH, MODEL_PATH, main  # noqa: E402


if __name__ == "__main__":
    print("=== FinAccess AI: automated retraining ===")
    print(f"Dataset: {DATA_PATH}")
    print(f"Model: {MODEL_PATH}")
    main()
    if MODEL_PATH.exists():
        print(f"Retraining completed. Model size: {MODEL_PATH.stat().st_size} bytes")
