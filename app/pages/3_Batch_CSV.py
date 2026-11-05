import pandas as pd
import streamlit as st

from utils.data import load_data
from utils.model import load_model, predict

st.title("Batch prediction from a CSV")

REQUIRED = ["date", "market", "commodity", "unit", "pricetype"]
COMBO = ["market", "commodity", "unit", "pricetype"]

df = load_data()
model = load_model()

if model is None:
    st.error("Model not found. Place the trained model in models/model_current.joblib.")
    st.stop()

# ---------- Reference tables built from the data ----------
markets_ref = df.drop_duplicates("market")[
    ["market", "admin1", "admin2", "latitude", "longitude"]
]
commodities_ref = df.drop_duplicates("commodity")[["commodity", "category"]]
known_combos = pd.MultiIndex.from_frame(df[COMBO].drop_duplicates())

# ---------- Template ----------
st.markdown(
    "Upload a CSV with these columns: "
    + ", ".join(f"`{c}`" for c in REQUIRED)
    + ". Department, coordinates and category are filled in automatically."
)
example = df.iloc[[-1]][REQUIRED].assign(date=lambda d: d["date"].dt.date)
st.download_button(
    "Download a CSV template",
    example.to_csv(index=False).encode("utf-8"),
    file_name="template_predictions.csv",
    mime="text/csv",
)

with st.expander("Valid product and market names"):
    names_left, names_right = st.columns(2)
    names_left.dataframe(
        pd.DataFrame({"product": sorted(commodities_ref["commodity"])}),
        hide_index=True,
    )
    names_right.dataframe(
        pd.DataFrame({"market": sorted(markets_ref["market"])}),
        hide_index=True,
    )

file = st.file_uploader("Your CSV file", type="csv")
if file is None:
    st.stop()

# ---------- Read and validate ----------
try:
    raw = pd.read_csv(file, sep=None, engine="python")  # detects , or ;
except Exception as exc:
    st.error(f"Could not read this file: {exc}")
    st.stop()

raw.columns = raw.columns.str.strip().str.lower()
missing = [c for c in REQUIRED if c not in raw.columns]
if missing:
    st.error(f"Missing column(s): {', '.join(missing)}")
    st.stop()

raw = raw[REQUIRED].copy()
for col in ["market", "commodity", "unit", "pricetype"]:
    raw[col] = raw[col].astype(str).str.strip()
raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
raw.insert(0, "csv_line", raw.index + 2)  # +2 = header line + 1-based counting

data = raw.merge(markets_ref, on="market", how="left").merge(
    commodities_ref, on="commodity", how="left"
)

reason = pd.Series("", index=data.index)
reason[data["date"].isna()] += "invalid date; "
reason[data["admin1"].isna()] += "unknown market; "
reason[data["category"].isna()] += "unknown product; "
valid_mask = reason == ""

valid = data[valid_mask].copy()
rejected = data[~valid_mask].assign(reason=reason[~valid_mask])

if valid.empty:
    st.error("No valid row to predict.")
    st.dataframe(rejected[["csv_line", *REQUIRED, "reason"]], width="stretch")
    st.stop()

# ---------- Predict ----------
valid["predicted_price_xof"] = predict(model, valid).round(0)
valid["known_combination"] = pd.MultiIndex.from_frame(valid[COMBO]).isin(known_combos)

# ---------- Display ----------
c1, c2, c3 = st.columns(3)
c1.metric("Rows predicted", f"{len(valid):,}")
c2.metric("Rows rejected", f"{len(rejected):,}")
c3.metric("Average predicted price", f"{valid['predicted_price_xof'].mean():,.0f} XOF")

n_unknown = int((~valid["known_combination"]).sum())
if n_unknown:
    st.warning(
        f"{n_unknown} row(s) combine a market and a product never observed together "
        "(known_combination = False). Treat these estimates with caution."
    )

out_cols = [*REQUIRED, "predicted_price_xof", "known_combination"]
result = valid[out_cols].assign(date=valid["date"].dt.date)
st.dataframe(result, width="stretch")

st.download_button(
    "Download the predictions",
    result.to_csv(index=False).encode("utf-8"),
    file_name="predictions.csv",
    mime="text/csv",
    type="primary",
)

if not rejected.empty:
    with st.expander(f"Rejected rows ({len(rejected)})"):
        st.dataframe(rejected[["csv_line", *REQUIRED, "reason"]], width="stretch")