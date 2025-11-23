import os
import streamlit as st
from utils.load_data import (
    load_price_data,
    load_prediction_data,
    load_prediction_history
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Downloads", page_icon="💾", layout="wide")
st.title("💾 Downloads")


# ============================================================
# LOAD DATA  (NO BASE_DIR NEEDED)
# ============================================================
price_df = load_price_data()
pred_df = load_prediction_data()
hist_df = load_prediction_history()


# ============================================================
# PRICE DATA DOWNLOAD
# ============================================================
st.subheader("📁 Price Data (stock_data.csv)")

if price_df.empty:
    st.warning("⚠ No price data found. Run `python run_daily.py` first.")
else:
    st.dataframe(price_df.tail(200), use_container_width=True, height=300)

    csv_price = price_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download Full Price Data CSV",
        data=csv_price,
        file_name="stock_data_export.csv",
        mime="text/csv",
    )


# ============================================================
# LATEST PREDICTIONS DOWNLOAD
# ============================================================
st.subheader("🤖 Latest AI Predictions (latest_predictions.csv)")

if pred_df.empty:
    st.warning("⚠ No predictions found. Run `python run_daily.py` first.")
else:
    st.dataframe(pred_df, use_container_width=True)

    csv_pred = pred_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download Latest Predictions CSV",
        data=csv_pred,
        file_name="latest_predictions_export.csv",
        mime="text/csv",
    )


# ============================================================
# PREDICTION HISTORY DOWNLOAD
# ============================================================
st.subheader("📚 Prediction History (predictions_history.csv)")

if hist_df.empty:
    st.info("No prediction history found yet. It will generate daily after running `run_daily.py`.")
else:
    st.dataframe(hist_df.tail(200), use_container_width=True, height=300)

    csv_hist = hist_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download Prediction History CSV",
        data=csv_hist,
        file_name="predictions_history_export.csv",
        mime="text/csv",
    )


# ============================================================
# FOOTER
# ============================================================
st.caption(
    """
    ✔ All files are generated automatically by **run_daily.py**  
    ✔ You can download them anytime  
    ✔ Streamlit Cloud-optimized (no BASE_DIR issues)
    """
)
