import streamlit as st
from utils.load_data import load_price_data
from utils.charts import (
    add_indicators,
    make_candlestick_with_sma,
    make_rsi_chart,
    make_macd_chart
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Charts", page_icon="📊", layout="wide")
st.title("📊 Pro Charts")

# ============================================================
# LOAD DATA  (NO BASE_DIR — auto-detected inside loader)
# ============================================================
df = load_price_data()

if df.empty:
    st.warning("⚠ No price data found. Run `python run_daily.py` first.")
    st.stop()

# ============================================================
# SYMBOL & LOOKBACK
# ============================================================
symbols = sorted(df["Symbol"].unique())

col1, col2 = st.columns(2)
with col1:
    symbol = st.selectbox("Select Symbol", symbols)

with col2:
    lookback_days = st.number_input(
        "Lookback days",
        min_value=30,
        max_value=5000,
        value=365,
        step=30
    )

# Filter selected symbol
sym_df = df[df["Symbol"] == symbol].sort_values("Date").tail(lookback_days)

# Add indicators
sym_df = add_indicators(sym_df)

# ============================================================
# BUILD CHARTS
# ============================================================
candle_fig = make_candlestick_with_sma(sym_df, symbol)
rsi_fig = make_rsi_chart(sym_df, symbol)
macd_fig = make_macd_chart(sym_df, symbol)

# ============================================================
# DISPLAY CHARTS
# ============================================================
st.plotly_chart(candle_fig, use_container_width=True)

col_rsi, col_macd = st.columns(2)
with col_rsi:
    st.plotly_chart(rsi_fig, use_container_width=True)
with col_macd:
    st.plotly_chart(macd_fig, use_container_width=True)
