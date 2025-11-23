import os
import joblib
import pandas as pd
import streamlit as st

from utils.load_data import (
    load_price_data,
    load_prediction_history,
)


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")
st.title("📈 Model Performance Dashboard")


# ============================================================
# LOAD PRICE DATA
# ============================================================
price_df = load_price_data()

if price_df.empty:
    st.warning("⚠ No price data found. Run `python run_daily.py` first.")
    st.stop()

col1, col2 = st.columns(2)
col1.metric("Symbols", len(price_df["Symbol"].unique()))
col2.metric("Total Rows", len(price_df))


# ============================================================
# LOAD MODELS (no BASE_DIR, auto-detected inside paths)
# ============================================================
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
MODEL_DIR = os.path.abspath(MODEL_DIR)

price_model_path = os.path.join(MODEL_DIR, "price_model.pkl")
dir_model_path = os.path.join(MODEL_DIR, "dir_model.pkl")

st.markdown("### 🔍 Installed Models")

if not (os.path.exists(price_model_path) and os.path.exists(dir_model_path)):
    st.warning("❌ Models not found. Run `python train_model.py` to train models.")
    st.stop()

price_model = joblib.load(price_model_path)
dir_model = joblib.load(dir_model_path)

st.write(f"**Price Model:** `{type(price_model).__name__}`")
st.write(f"**Direction Model:** `{type(dir_model).__name__}`")


# ============================================================
# LOAD PREDICTION HISTORY (if available)
# ============================================================
hist_df = load_prediction_history()

st.markdown("---")
st.subheader("📚 Historical Prediction Accuracy")

if hist_df.empty:
    st.info(
        """
        No prediction history found.

        After you run **run_daily.py**, a file  
        `predictions_history.csv` will automatically update every day.

        Come back later to see rolling accuracy, confusion matrix, and performance charts.
        """
    )
    st.stop()


# ============================================================
# CLEAN & PREPARE HISTORY
# ============================================================
hist_df = hist_df.dropna(subset=["Predicted_Direction", "Probability_Up"])

# merge with true prices
merged = pd.merge(
    hist_df,
    price_df,
    on=["Date", "Symbol"],
    how="inner"
)

merged["True_Direction"] = (
    merged["Close"].shift(-1) > merged["Close"]
).astype(int)

merged["Pred_Class"] = (merged["Predicted_Direction"] == "UP").astype(int)


# ============================================================
# BASIC METRICS
# ============================================================
accuracy = (merged["Pred_Class"] == merged["True_Direction"]).mean()

colA, colB = st.columns(2)
colA.metric("Overall Direction Accuracy", f"{accuracy*100:.2f}%")
colB.metric("History Rows", len(merged))


# ============================================================
# SHOW TABLE
# ============================================================
st.markdown("### 📄 Full Prediction + Actual Comparison")
st.dataframe(merged.sort_values("Date", ascending=False), use_container_width=True)


# ============================================================
# INFO FOOTER
# ============================================================
st.info(
    """
    ✔ This dashboard will become richer as more history accumulates.  
    ✔ It will later include:  
    - Rolling accuracy  
    - Per-symbol heatmaps  
    - Confusion matrix  
    - Win-rate statistics  
    - Drift detection  

    Keep running `python run_daily.py` every day to build history.
    """
)
