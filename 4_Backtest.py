import streamlit as st
from utils.load_data import load_price_data
from utils.strategy import atr_strategy_backtest
import plotly.express as px


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Backtest", page_icon="🧪", layout="wide")
st.title("🧪 ATR-based Strategy Backtest (Example)")


# ============================================================
# LOAD PRICE DATA (NO BASE_DIR — auto-detected inside loader)
# ============================================================
df = load_price_data()

if df.empty:
    st.warning("⚠ No price data found. Run `python run_daily.py` first.")
    st.stop()


# ============================================================
# USER INPUTS
# ============================================================
symbols = sorted(df["Symbol"].unique())

col1, col2 = st.columns(2)
with col1:
    symbol = st.selectbox("Select Symbol", symbols)

with col2:
    lookback_days = st.number_input(
        "Lookback days",
        min_value=100,
        max_value=5000,
        value=365,
        step=50
    )

# Filter data
sym_df = df[df["Symbol"] == symbol].sort_values("Date").tail(lookback_days)


# ============================================================
# RUN BACKTEST
# ============================================================
result = atr_strategy_backtest(sym_df)

if result is None:
    st.warning("⚠ Not enough data to run ATR strategy.")
    st.stop()

bt_df = result["df"]
total_return = result["total_return"]
max_dd = result["max_drawdown"]


# ============================================================
# DISPLAY METRICS
# ============================================================
colA, colB = st.columns(2)
colA.metric("Total Return", f"{total_return*100:.1f}%")
colB.metric("Max Drawdown", f"{max_dd*100:.1f}%")


# ============================================================
# PLOT EQUITY CURVE
# ============================================================
fig = px.line(
    bt_df,
    x="Date",
    y="Equity",
    title=f"{symbol} – Equity Curve (ATR Strategy)"
)
st.plotly_chart(fig, use_container_width=True)


# ============================================================
# FOOTER
# ============================================================
st.caption(
    """
    This is a **technical ATR-based example strategy**, independent of the ML signals.

    To backtest the **actual AI predictions strategy**,  
    you would need the full `predictions_history.csv` generated daily by:
    `python run_daily.py`.
    """
)
