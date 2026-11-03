import streamlit as st

st.set_page_config(page_title="FoodPrice AI", layout="wide")

pages = [
    st.Page("Home.py", title="Home", icon=":material/home:", default=True),
    st.Page("pages/1_Dashboard.py", title="Dashboard", icon=":material/monitoring:"),
    st.Page("pages/2_Estimation.py", title="Estimation", icon=":material/sell:"),
    st.Page("pages/3_Batch_CSV.py", title="Batch CSV", icon=":material/upload_file:"),
]
st.navigation(pages).run()