from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"

# Columns kept for the dashboard and the app
KEEP_COLUMNS = [
    "date", "admin1", "admin2", "market", "market_id",
    "latitude", "longitude", "category", "commodity", "commodity_id",
    "unit", "pricetype", "price",
]


@st.cache_data(show_spinner="Loading data...")
def load_data() -> pd.DataFrame:
    """Load all parquet files, merge them and add time features."""
    frames = []
    for name in DATA_FILES:
        path = DATA_DIR / name
        if path.exists():
            frames.append(pd.read_parquet(path))

    if not frames:
        st.error(f"No parquet file found in {DATA_DIR}")
        st.stop()

    df = pd.concat(frames, ignore_index=True)
    df = df[KEEP_COLUMNS].drop_duplicates()
    df = df.dropna(subset=["price", "date"])
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df.sort_values("date").reset_index(drop=True)


def get_options(df: pd.DataFrame) -> dict:
    """Sorted lists used to fill the select boxes."""
    return {
        "departments": sorted(df["admin1"].unique()),
        "categories": sorted(df["category"].unique()),
        "commodities": sorted(df["commodity"].unique()),
        "markets": sorted(df["market"].unique()),
        "units": sorted(df["unit"].unique()),
        "pricetypes": sorted(df["pricetype"].unique()),
        "min_date": df["date"].min().date(),
        "max_date": df["date"].max().date(),
    }
