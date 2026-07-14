"""
app.py
Main Streamlit dashboard entry point.
Integrates predictions from all four modules: Boiler, Throughput, Energy, Yield.
"""

import streamlit as st

st.set_page_config(page_title="Refinery Performance Optimization", layout="wide")

st.title("AI-Powered Refinery Performance Optimization System")
st.markdown("Predict boiler efficiency, throughput, energy consumption, and product yield.")

module = st.sidebar.selectbox(
    "Select Module",
    ["Boiler Efficiency", "CDU Throughput", "Energy Consumption", "Product Yield"],
)

st.header(module)
st.info("TODO: connect this page to the trained model in `models/` via `src/predict.py`.")

# TODO:
# - Load relevant model with src.predict.load_model()
# - Add input widgets for process parameters
# - Display prediction + SHAP explanation plot
