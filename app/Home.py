import streamlit as st

from utils.data import get_options, load_data

st.set_page_config(page_title="FoodPrice AI", page_icon="🌽", layout="wide")

st.title("FoodPrice AI")
st.caption("Analysis and price prediction of food products on Benin markets")

df = load_data()
opts = get_options(df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Observations", f"{len(df):,}")
c2.metric("Products", len(opts["commodities"]))
c3.metric("Markets", len(opts["markets"]))
c4.metric("Period", f"{opts['min_date'].year} to {opts['max_date'].year}")

st.dataframe(df.head(20), width="stretch")
