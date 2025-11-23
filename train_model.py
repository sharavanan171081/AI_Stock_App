import os
import joblib
import numpy as np
import pandas as pd
import ta

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


# ============================================================
# AUTO-DETECT PROJECT ROOT (WORKS ON CLOUD + WINDOWS)
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "stock_data.csv"
MODEL_DIR = BASE_DIR / "model"

MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def create_features(df: pd.DataFrame):

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")

    # Ensure numeric
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    frames = []

    for symbol, g in df.groupby("Symbol"):
        g = g.sort_values("Date").copy()

        # SMAs
        g["SMA_5"] = g["Close"].rolling(5).mean()
        g["SMA_10"] = g["Close"].rolling(10).mean()
        g["SMA_20"] = g["Close"].rolling(20).mean()

        # RSI
        try:
            g["RSI_14"] = ta.momentum.rsi(g["Close"], window=14)
        except Exception:
            g["RSI_14"] = np.nan

        # MACD
        try:
            macd = ta.trend.macd(g["Close"], window_fast=12, window_slow=26)
            macd_signal = ta.trend.macd_signal(
                g["Close"], window_fast=12, window_slow=26, window_sign=9
            )
            g["MACD"] = macd
            g["MACD_SIGNAL"] = macd_signal
        except Exception:
            g["MACD"] = np.nan
            g["MACD_SIGNAL"] = np.nan

        # Bollinger
        try:
            bb = ta.volatility.BollingerBands(g["Close"])
            g["BB_HIGH"] = bb.bollinger_hband()
            g["BB_LOW"] = bb.bollinger_lband()
        except Exception:
            g["BB_HIGH"] = np.nan
            g["BB_LOW"] = np.nan

        # ATR
        try:
            g["ATR_14"] = ta.volatility.average_true_range(
                g["High"], g["Low"], g["Close"], window=14
            )
        except Exception:
            g["ATR_14"] = np.nan

        # Returns & volatility
        g["Ret_1d"] = g["Close"].pct_change(1)
        g["Ret_5d"] = g["Close"].pct_change(5)
        g["Vol_Change"] = g["Volume"].pct_change(1)
        g["Rolling_Volatility_10"] = g["Ret_1d"].rolling(10).std()

        # Targets
        g["Next_Close"] = g["Close"].shift(-1)
        g["Direction"] = (g["Next_Close"] > g["Close"]).astype(int)

        frames.append(g)

    df_feat = pd.concat(frames)

    feature_cols = [
        "Close", "SMA_5", "SMA_10", "SMA_20",
        "RSI_14", "MACD", "MACD_SIGNAL",
        "BB_HIGH", "BB_LOW", "ATR_14",
        "Ret_1d", "Ret_5d", "Vol_Change",
        "Rolling_Volatility_10",
        "Volume"
    ]

    # Remove bad rows
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
    df_feat = df_feat.dropna(subset=feature_cols + ["Next_Close"])

    return df_feat, feature_cols


# ============================================================
# TRAINING PIPELINE
# ============================================================
def main():

    if not DATA_FILE.exists():
        print(f"❌ Data file not found: {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE)
    print(f"📄 Loaded {len(df)} rows from {DATA_FILE}")

    df_feat, feature_cols = create_features(df)

    # Clean again after merge
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
    df_feat = df_feat.dropna(subset=feature_cols + ["Next_Close"])

    # ML Inputs
    X = df_feat[feature_cols]
    y_price = df_feat["Next_Close"]
    y_dir = df_feat["Direction"]

    # Train-test split
    X_train, X_test, y_price_train, y_price_test, y_dir_train, y_dir_test = train_test_split(
        X, y_price, y_dir, test_size=0.20, shuffle=False
    )

    # =====================
    # PRICE MODEL
    # =====================
    price_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=14,
        random_state=42,
        n_jobs=-1
    )
    price_model.fit(X_train, y_price_train)

    # =====================
    # DIRECTION MODEL
    # =====================
    dir_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        random_state=42,
        n_jobs=-1
    )
    dir_model.fit(X_train, y_dir_train)

    # Evaluate
    price_r2 = price_model.score(X_test, y_price_test)
    dir_acc = dir_model.score(X_test, y_dir_test)

    print(f"✅ Price model R² (test): {price_r2:.3f}")
    print(f"✅ Direction model Accuracy (test): {dir_acc:.3f}")

    # Save models
    price_path = MODEL_DIR / "price_model.pkl"
    dir_path = MODEL_DIR / "dir_model.pkl"

    joblib.dump(price_model, price_path)
    joblib.dump(dir_model, dir_path)

    print(f"💾 Saved price model → {price_path}")
    print(f"💾 Saved direction model → {dir_path}")


# ============================================================
# ENTRY
# ============================================================
if __name__ == "__main__":
    main()
