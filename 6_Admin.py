import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Admin", page_icon="🛠️", layout="wide")
st.title("🛠️ Admin & System Notes")

st.markdown(
    """
    ### 🧰 System Overview

    This Admin page provides instructions to manage:

    - Daily data fetching  
    - Model retraining  
    - Manual maintenance commands  
    - Deployment notes for Streamlit Cloud  

    ⚠ **Important:**  
    Streamlit Cloud cannot run your Python scripts (`run_daily.py`, `train_model.py`)
    automatically. They must be executed on your **local PC or server**.
    """
)

# ============================================================
# COMMANDS SECTION
# ============================================================
st.subheader("📦 Manual Commands")

st.code(
    """
# Fetch full historical NSE data
python nse_fetch.py

# Retrain ML models (price + direction)
python train_model.py

# Generate today's predictions + append to history
python run_daily.py

# Launch dashboard locally
streamlit run app.py
""",
    language="bash",
)

# ============================================================
# CLOUD INFO
# ============================================================
st.subheader("☁ Deployment Notes")

st.markdown(
    """
- Upload the full project to a **GitHub repository**  
- Deploy on **Streamlit Cloud** using your repo  
- Streamlit will run `app.py` automatically  
- Data & model folders must be included in your repo  

To automate daily predictions, use **Windows Task Scheduler** (or Linux cron):

