import streamlit as st
import pandas as pd

# Correct imports – NO BASE_DIR parameter needed
from utils.load_data import load_price_data, load_prediction_data


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Stock Trading Desk",
    page_icon="📈",
    layout="wide",
)

st.title("🧠 AI Stock Trading Desk")

st.markdown(
    """
    Welcome to your **institutional-grade AI trading dashboard** for Indian equities.

    Navigate using the sidebar:

    - 📊 **Predictions** – Today's AI signals  
    - 🕯 **Charts** – Candlesticks + Indicators  
    - 📈 **Performance** – Accuracy & analytics  
    - 🧪 **Backtest** – ATR & strategy simulation  
    - 📥 **Downloads** – Export data  
    - ⚙ **Admin** – Maintenance + tools  
    """
)


# ============================================================
# LOAD DATA
# BASE_DIR is auto-detected INSIDE load_data.py
# ============================================================
price_df = load_price_data()
pred_df = load_prediction_data()


# ============================================================
# SUMMARY STATISTICS
# ============================================================
col1, col2 = st.columns(2)

with col1:
    if price_df.empty:
        st.error("❌ No price data found. Run `python run_daily.py` first.")
    else:
        st.metric("Symbols in system", len(price_df["Symbol"].unique()))
        st.metric("Total history rows", len(price_df))

with col2:
    if pred_df.empty:
        st.metric("Stocks with AI prediction today", 0)
    else:
        st.metric("Stocks with AI prediction today", len(pred_df))

        best = pred_df.sort_values("Probability_Up", ascending=False).iloc[0]
        st.markdown(
            f"**🔥 Top Bullish Signal:** `{best['Symbol']}` "
            f"→ **{best['Probability_Up']*100:.1f}% UP**"
        )


# ============================================================
# FOOTER INFO
# ============================================================
st.info(
    """
    ✔ Run `python run_daily.py` daily to refresh predictions  
    ✔ Run `python train_model.py` weekly or monthly to retrain models  
    ✔ All files load automatically (no hard-coded paths)  
    """
)
