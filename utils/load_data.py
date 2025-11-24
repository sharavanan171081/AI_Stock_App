import os
from pathlib import Path
import pandas as pd
import streamlit as st


# ============================================================
#   AUTO-DETECT BASE DIRECTORY (WORKS ON ANY COMPUTER / CLOUD)
# ============================================================
def get_base_dir() -> Path:
    """
    BASE_DIR will always be the project root.
    Works on:
        • Local Windows (E:\)
        • Streamlit Cloud
        • Linux servers
        • Any relative environment
    """
    return Path(__file__).resolve().parents[1]


# ============================================================
#   LOAD HISTORICAL PRICE DATA (stock_data.csv)
# ============================================================
@st.cache_data
def load_price_data() -> pd.DataFrame:
    base_dir = get_base_dir()
    path = base_dir / "data" / "stock_data.csv"

    if not path.exists():
        st.error(f"❌ Prices file not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)

    # Convert Date column
    # stock_data.csv uses DD-MM-YYYY format
    try:
        df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    except:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Convert numeric fields
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    return df


# ============================================================
#   LOAD LATEST PREDICTIONS (latest_predictions.csv)
# ============================================================
@st.cache_data
def load_prediction_data() -> pd.DataFrame:
    base_dir = get_base_dir()
    path = base_dir / "data" / "latest_predictions.csv"

    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    # latest_predictions.csv uses YYYY-MM-DD format
    try:
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    except:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Clean numeric fields
    if "Probability_Up" in df.columns:
        df["Probability_Up"] = pd.to_numeric(df["Probability_Up"], errors="coerce")

    if "Predicted_Price" in df.columns:
        df["Predicted_Price"] = pd.to_numeric(df["Predicted_Price"], errors="coerce")

    return df


# ============================================================
#   LOAD PREDICTION HISTORY (predictions_history.csv)
# ============================================================
@st.cache_data
def load_prediction_history() -> pd.DataFrame:
    base_dir = get_base_dir()
    path = base_dir / "data" / "predictions_history.csv"

    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    # History file uses YYYY-MM-DD
    try:
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    except:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    if "Run_Timestamp" in df.columns:
        df["Run_Timestamp"] = pd.to_datetime(df["Run_Timestamp"], errors="coerce")

    # Clean numeric fields
    numeric_cols = ["Predicted_Price", "Probability_Up"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
