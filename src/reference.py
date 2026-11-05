"""Table de référence pour l'application (Personne 3).

L'utilisateur choisit un produit, un marché et une date. Le modèle a besoin de 10
colonnes : ce module retrouve les autres (département, commune, coordonnées,
catégorie, unité) à partir de ce choix.

    ref = load_reference()
    row = make_input_row(ref, market="Cotonou", commodity="Maize", date="2026-03-15")
    prix = model.predict(row)[0]
"""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

from src.schema import INPUT_COLUMNS

ROOT = Path(__file__).resolve().parent.parent
REFERENCE_PATH = ROOT / "reference.json"


def _mode(s: pd.Series):
    return s.mode().iloc[0]


def build_reference(df: pd.DataFrame) -> dict:
    """Construit la table de référence à partir d'un DataFrame nettoyé."""
    markets = {}
    for market, g in df.groupby("market"):
        markets[market] = {
            "admin1": _mode(g["admin1"]), "admin2": _mode(g["admin2"]),
            "latitude": float(g["latitude"].median()),
            "longitude": float(g["longitude"].median()),
        }
    commodities = {}
    for commodity, g in df.groupby("commodity"):
        commodities[commodity] = {
            "category": _mode(g["category"]),
            "units": g["unit"].value_counts().index.tolist(),  # unité la plus fréquente d'abord
        }
    return {
        "markets": markets,
        "commodities": commodities,
        "pricetypes": sorted(df["pricetype"].unique().tolist()),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
    }


def save_reference(ref: dict, path=REFERENCE_PATH) -> None:
    Path(path).write_text(json.dumps(ref, ensure_ascii=False, indent=1), encoding="utf-8")


def load_reference(path=REFERENCE_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_input_row(ref: dict, market: str, commodity: str, date,
                   unit: str | None = None, pricetype: str = "Retail") -> pd.DataFrame:
    """Une ligne prête pour model.predict() à partir du choix de l'utilisateur."""
    if market not in ref["markets"]:
        raise KeyError(f"Marché inconnu : {market}")
    if commodity not in ref["commodities"]:
        raise KeyError(f"Produit inconnu : {commodity}")
    m, c = ref["markets"][market], ref["commodities"][commodity]
    row = {
        "date": pd.to_datetime(date), "admin1": m["admin1"], "admin2": m["admin2"],
        "market": market, "category": c["category"], "commodity": commodity,
        "unit": unit or c["units"][0], "pricetype": pricetype,
        "latitude": m["latitude"], "longitude": m["longitude"],
    }
    return pd.DataFrame([row], columns=INPUT_COLUMNS)
