import os
import numpy as np
import pandas as pd
import joblib
import ta
from datetime import datetime
from pathlib import Path

import nse_fetch   # Fetches & rebuilds stock_data.csv automatically


# ============================================================
#  AUTO-DETECT PROJECT ROOT  (WORKS ON CLOUD + WINDOWS)
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "stock_data.csv"
PRED_FILE = BASE_DIR / "data" / "latest_predictions.csv"
HISTORY_FILE = BASE_DIR / "data" / "predictions_history.csv"
PRICE_MODEL_FILE = BASE_DIR / "model" / "price_model.pkl"
DIR_MODEL_FILE = BASE_DIR / "model" / "dir_model.pkl"


# ============================================================
#  FEATURE ENGINEERING FOR PREDICTION
# ============================================================
def create_features_for_prediction(df: pd.DataFrame):

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")

    # Ensure numeric
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    frames = []

    for symbol, g in df.groupby("Symbol"):
        g = g.sort_values("Date").copy()

        # Indicators
        g["SMA_5"] = g["Close"].rolling(5).mean()
        g["SMA_10"] = g["Close"].rolling(10).mean()
        g["SMA_20"] = g["Close"].rolling(20).mean()

        # RSI
        g["RSI_14"] = ta.momentum.rsi(g["Close"], window=14)

        # MACD
        macd = ta.trend.macd(g["Close"])
        g["MACD"] = macd
        g["MACD_SIGNAL"] = ta.trend.macd_signal(g["Close"])

        # Bollinger
        bb = ta.volatility.BollingerBands(g["Close"])
        g["BB_HIGH"] = bb.bollinger_hband()
        g["BB_LOW"] = bb.bollinger_lband()

        # ATR
        g["ATR_14"] = ta.volatility.average_true_range(
            g["High"], g["Low"], g["Close"]
        )

        # Returns & Volatility
        g["Ret_1d"] = g["Close"].pct_change(1)
        g["Ret_5d"] = g["Close"].pct_change(5)
        g["Vol_Change"] = g["Volume"].pct_change(1)
        g["Rolling_Volatility_10"] = g["Ret_1d"].rolling(10).std()

        feature_cols = [
            "Close", "SMA_5", "SMA_10", "SMA_20", "RSI_14",
            "MACD", "MACD_SIGNAL", "BB_HIGH", "BB_LOW",
            "ATR_14", "Ret_1d", "Ret_5d", "Vol_Change",
            "Rolling_Volatility_10", "Volume"
        ]

        g = g.dropna(subset=feature_cols)

        if not g.empty:
            frames.append(g.tail(1))

    if not frames:
        return pd.DataFrame(), []

    df_last = pd.concat(frames).reset_index(drop=True)
    return df_last, feature_cols


# ============================================================
#  MAIN PIPELINE
# ============================================================
def main():

    print("📥 Fetching latest NSE data...")
    nse_fetch.main()   # Auto rebuild stock_data.csv

    # Ensure models exist
    if not PRICE_MODEL_FILE.exists() or not DIR_MODEL_FILE.exists():
        print("❌ Model files missing. Run: python train_model.py")
        return

    # Load models
    print("📄 Loading models...")
    price_model = joblib.load(PRICE_MODEL_FILE)
    dir_model = joblib.load(DIR_MODEL_FILE)

    # Load stock data
    df = pd.read_csv(DATA_FILE)
    print(f"📄 Loaded {len(df)} rows for prediction")

    # Create features
    df_feat, feature_cols = create_features_for_prediction(df)
    if df_feat.empty:
        print("❌ Not enough data for indicators.")
        return

    # Clean infinite & NaN
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan).dropna()

    # Prepare ML inputs
    X = df_feat[feature_cols]

    # Predict
    price_preds = price_model.predict(X)
    dir_probs = dir_model.predict_proba(X)[:, 1]

    # Build output
    results = []
    for i, row in df_feat.iterrows():

        symbol = row["Symbol"]
        date_str = row["Date"].strftime("%Y-%m-%d")

        prob_up = float(dir_probs[i])
        predicted_price = float(price_preds[i])
        direction = "UP" if prob_up >= 0.5 else "DOWN"

        results.append({
            "Date": date_str,
            "Symbol": symbol,
            "Predicted_Price": round(predicted_price, 2),
            "Predicted_Direction": direction,
            "Probability_Up": round(prob_up, 4)
        })

    pred_df = pd.DataFrame(results)

    # Save latest predictions
    pred_df.to_csv(PRED_FILE, index=False)
    print(f"✅ Saved today's predictions to: {PRED_FILE}")
    print(pred_df)

    # ============================================================
    # Append to HISTORY FILE
    # ============================================================
    pred_df_hist = pred_df.copy()
    pred_df_hist["Run_Timestamp"] = datetime.now()

    if HISTORY_FILE.exists():
        old = pd.read_csv(HISTORY_FILE)
        combined = pd.concat([old, pred_df_hist], ignore_index=True)
        combined = combined.drop_duplicates(["Date", "Symbol"], keep="last")
        combined.to_csv(HISTORY_FILE, index=False)
    else:
        pred_df_hist.to_csv(HISTORY_FILE, index=False)

    print(f"🕒 Prediction history updated: {HISTORY_FILE}")


# ENTRY POINT
if __name__ == "__main__":
    main()
