import plotly.express as px
import streamlit as st

from utils.data import get_options, load_data

st.title("Price dashboard")

df = load_data()
opts = get_options(df)

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")

categories = st.sidebar.multiselect("Category (optional)", opts["categories"])
scope = df[df["category"].isin(categories)] if categories else df

commodity = st.sidebar.selectbox("Product", sorted(scope["commodity"].unique()))
product_df = scope[scope["commodity"] == commodity]

# Prices are only comparable for the same unit and price type
unit = st.sidebar.selectbox("Unit", sorted(product_df["unit"].unique()))
product_df = product_df[product_df["unit"] == unit]

pricetype = st.sidebar.selectbox("Price type", sorted(product_df["pricetype"].unique()))
product_df = product_df[product_df["pricetype"] == pricetype]

departments = st.sidebar.multiselect(
    "Department (optional)", sorted(product_df["admin1"].unique())
)
if departments:
    product_df = product_df[product_df["admin1"].isin(departments)]

date_range = st.sidebar.date_input(
    "Period",
    value=(product_df["date"].min().date(), product_df["date"].max().date()),
    min_value=opts["min_date"],
    max_value=opts["max_date"],
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    product_df = product_df[
        (product_df["date"].dt.date >= start) & (product_df["date"].dt.date <= end)
    ]

if product_df.empty:
    st.warning("No data for this selection. Try changing the filters.")
    st.stop()

# ---------- KPIs ----------
st.subheader(f"{commodity} (XOF per {unit}, {pricetype.lower()})")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Observations", f"{len(product_df):,}")
k2.metric("Average price", f"{product_df['price'].mean():,.0f} XOF")
k3.metric("Median price", f"{product_df['price'].median():,.0f} XOF")
k4.metric("Markets", product_df["market"].nunique())

# ---------- Charts ----------
tab_time, tab_market, tab_dept = st.tabs(["Evolution", "Markets", "Departments"])

with tab_time:
    by_dept = st.checkbox("Split by department")
    product_df = product_df.assign(
        period=product_df["date"].dt.to_period("M").dt.to_timestamp()
    )
    group_cols = ["period", "admin1"] if by_dept else ["period"]
    trend = product_df.groupby(group_cols, as_index=False)["price"].mean()
    fig = px.line(
        trend,
        x="period",
        y="price",
        color="admin1" if by_dept else None,
        markers=True,
        labels={"period": "Month", "price": "Average price (XOF)", "admin1": "Department"},
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_market:
    top_n = st.slider("Number of markets", 5, 30, 15)
    markets = (
        product_df.groupby("market", as_index=False)["price"]
        .mean()
        .sort_values("price", ascending=False)
        .head(top_n)
    )
    fig = px.bar(
        markets.sort_values("price"),
        x="price",
        y="market",
        orientation="h",
        labels={"price": "Average price (XOF)", "market": "Market"},
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_dept:
    fig = px.box(
        product_df,
        x="admin1",
        y="price",
        labels={"admin1": "Department", "price": "Price (XOF)"},
    )
    st.plotly_chart(fig, use_container_width=True)

with st.expander("Show raw data"):
    st.dataframe(
        product_df.drop(columns=["period"]).sort_values("date", ascending=False),
        use_container_width=True,
    )