import streamlit as st
from utils.load_data import load_price_data, load_prediction_data


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Predictions",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Predictions – Today")


# ============================================================
# LOAD DATA (NO BASE_DIR PARAMETER ANYMORE)
# ============================================================
price_df = load_price_data()
pred_df = load_prediction_data()

if pred_df.empty:
    st.warning("⚠ No predictions file found yet. Run `python run_daily.py` first.")
    st.stop()


# ============================================================
# SYMBOL SELECTION
# ============================================================
symbols = sorted(pred_df["Symbol"].unique())

col1, col2 = st.columns(2)
with col1:
    selected_symbol = st.selectbox("Select Symbol", symbols)

with col2:
    auto_refresh = st.checkbox("Auto-refresh (every 60 sec)", value=False)

if auto_refresh:
    st.experimental_rerun()


# ============================================================
# SELECT SYMBOL DETAILS
# ============================================================
sym_pred = pred_df[pred_df["Symbol"] == selected_symbol].iloc[0]
sym_price = price_df[price_df["Symbol"] == selected_symbol].sort_values("Date")

last_close = sym_price.iloc[-1]["Close"]


# ============================================================
# METRIC CARDS
# ============================================================
colA, colB, colC, colD = st.columns(4)

colA.metric("Last Close", f"{last_close:,.2f}")
colB.metric("Predicted Price (Next Day)", f"{sym_pred['Predicted_Price']:,.2f}")
colC.metric("Probability UP (%)", f"{sym_pred['Probability_Up']*100:.1f}%")
colD.metric("Direction", sym_pred["Predicted_Direction"])


# ============================================================
# FULL PREDICTION TABLE
# ============================================================
st.subheader("📊 All Current AI Signals")

st.dataframe(
    pred_df.sort_values("Probability_Up", ascending=False),
    use_container_width=True,
)
