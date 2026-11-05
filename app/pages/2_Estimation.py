from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data import get_options, load_data
from utils.model import load_model, predict

st.title("Price estimation")

df = load_data()
opts = get_options(df)
model = load_model()

if model is None:
    st.error("Model not found. Place the trained model in models/model_current.joblib.")
    st.stop()

# ---------- Inputs (each choice narrows the next one) ----------
left, right = st.columns(2)

with left:
    commodity = st.selectbox("Product", opts["commodities"])
    sub = df[df["commodity"] == commodity]
    unit = st.selectbox("Unit", sorted(sub["unit"].unique()))
    sub = sub[sub["unit"] == unit]
    pricetype = st.selectbox("Price type", sorted(sub["pricetype"].unique()))
    sub = sub[sub["pricetype"] == pricetype]

with right:
    market = st.selectbox("Market", sorted(sub["market"].unique()))
    target_date = st.date_input(
        "Date", value=date.today(), min_value=opts["min_date"]
    )

history = sub[sub["market"] == market].sort_values("date")
info = history.iloc[-1]  # location and category come from the data

if target_date > opts["max_date"]:
    st.info(
        f"Your date is after the last observation ({opts['max_date']}). "
        "Tree-based models cannot extrapolate trends, so treat this as indicative."
    )

# ---------- Prediction ----------
if st.button("Estimate price", type="primary"):
    row = pd.DataFrame([{
        "admin1": info["admin1"],
        "admin2": info["admin2"],
        "market": market,
        "latitude": info["latitude"],
        "longitude": info["longitude"],
        "category": info["category"],
        "commodity": commodity,
        "unit": unit,
        "pricetype": pricetype,
        "date": pd.Timestamp(target_date),
    }])
    estimate = float(predict(model, row)[0])

    last = history.iloc[-1]
    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated price", f"{estimate:,.0f} XOF per {unit}")
    m2.metric(
        "Last observed price",
        f"{last['price']:,.0f} XOF",
        delta=f"estimate minus last: {estimate - last['price']:+,.0f} XOF",
        delta_color="off",
    )
    m3.metric("Historical average", f"{history['price'].mean():,.0f} XOF")

    # Context chart: history for this exact product, market, unit and price type
    fig = px.line(
        history, x="date", y="price", markers=True,
        labels={"date": "Date", "price": "Price (XOF)"},
        title=f"{commodity} at {market}: observed prices",
    )
    fig.add_scatter(
        x=[pd.Timestamp(target_date)], y=[estimate],
        mode="markers", name="Estimate",
        marker=dict(size=14, symbol="star", color="red"),
    )
    st.plotly_chart(fig, width="stretch")
