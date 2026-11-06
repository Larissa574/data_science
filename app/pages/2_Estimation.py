from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data import get_options, load_data
from utils.model import FEATURES, load_metadata, load_model, predict

st.title("Price estimation")

df = load_data()
opts = get_options(df)
model = load_model()

if model is None:
    st.error("Model not found. Place the trained model in models/model_current.joblib.")
    st.stop()

meta = load_metadata()
if meta:
    if set(meta.get("input_columns", FEATURES)) != set(FEATURES):
        st.warning("This model expects different input columns than the app sends.")
    with st.expander("About the model"):
        st.caption(
            f"{meta.get('model_label', 'Model')}, version {meta.get('version', '?')}, "
            f"trained on {str(meta.get('trained_at', ''))[:10]}"
        )
        m = meta.get("metrics_test") or meta.get("metrics_validation")
        if m:
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE", f"{m['MAE']:,.0f} XOF")
            c2.metric("RMSE", f"{m['RMSE']:,.0f} XOF")
            c3.metric("R²", f"{m['R2']:.3f}")


# ---------- Inputs written as a sentence (each choice narrows the next one) ----------
def say(col, text):
    """Write a piece of the sentence in larger text, next to the dropdowns."""
    col.markdown(f"<span style='font-size:1.2rem'>{text}</span>", unsafe_allow_html=True)


with st.container(border=True):
    st.caption("Describe the price you want to estimate")

    row1 = st.columns([2.2, 3, 0.8, 1.6, 4], vertical_alignment="center")
    say(row1[0], "I want the price of")
    with row1[1]:
        commodity = st.selectbox(
            "Product", opts["commodities"], label_visibility="collapsed"
        )
    sub = df[df["commodity"] == commodity]
    say(row1[2], "per")
    with row1[3]:
        unit = st.selectbox(
            "Unit", sorted(sub["unit"].unique()), label_visibility="collapsed"
        )
    sub = sub[sub["unit"] == unit]

    row2 = st.columns([0.6, 1.6, 2.6, 3, 0.8, 2.2], vertical_alignment="center")
    say(row2[0], "at")
    with row2[1]:
        pricetype = st.selectbox(
            "Price type", sorted(sub["pricetype"].unique()), label_visibility="collapsed"
        )
    sub = sub[sub["pricetype"] == pricetype]
    say(row2[2], "price, in the market of")
    with row2[3]:
        market = st.selectbox(
            "Market", sorted(sub["market"].unique()), label_visibility="collapsed"
        )
    say(row2[4], "on")
    with row2[5]:
        target_date = st.date_input(
            "Date",
            value=opts["max_date"],
            min_value=opts["min_date"],
            label_visibility="collapsed",
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
    with st.container(border=True):
        m1, m2, m3 = st.columns([1.5, 1, 1], vertical_alignment="center")
        m1.markdown("**Estimated price**")
        m1.markdown(
            f"<span style='font-size:2.8rem;font-weight:700'>{estimate:,.0f} XOF</span>"
            f"<span style='font-size:1.1rem'> per {unit}</span>",
            unsafe_allow_html=True,
        )
        m2.metric(
            "Last observed price",
            f"{last['price']:,.0f} XOF",
            delta=f"{estimate - last['price']:+,.0f} XOF vs last observed",
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
        mode="markers+text", name="Estimate",
        text=[f"{estimate:,.0f} XOF"], textposition="top center",
        marker=dict(size=14, symbol="star", color="red"),
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", y=1.12, x=0),
    )
    st.plotly_chart(fig, width="stretch")