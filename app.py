import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap
import time

# -----------------------------
# Load Models (Updated Paths)
# -----------------------------
@st.cache_resource
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# Scenario generator (VAE or custom)
scenario_generator = load_pickle("models/deployment/scenario_generator.pkl")

# Financial forecasting models
xgb_models = {
    "Revenue": load_pickle("models/deployment/xgboost_revenue.pkl"),
    "EPS": load_pickle("models/deployment/xgboost_eps.pkl"),
    "Profit Margin": load_pickle("models/deployment/xgboost_profit_margin.pkl"),
    "Debt/Equity": load_pickle("models/deployment/xgboost_debt_equity.pkl"),
    "Stock Return": load_pickle("models/deployment/xgboost_stock_return.pkl"),
}

# Risk detection model
risk_model = load_pickle("models/deployment/risk_model.pkl")
