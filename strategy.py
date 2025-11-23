import pandas as pd
import numpy as np
import ta


def atr_strategy_backtest(
    df: pd.DataFrame,
    atr_mult_stop: float = 2.0,
    atr_mult_tp: float = 3.0
):
    """
    ATR Trend Strategy Backtest
    ----------------------------------
    This strategy simulates a long-only trend system using:
        • SMA 20 trend filter
        • RSI 14 momentum filter
        • ATR stop-loss and take-profit
        
    Entry:
        Long when:
            - Close > SMA20
            - RSI > 50

    Exit Conditions:
        - Stop-loss hit  → Close <= Entry - ATR * stop_mult
        - Take-profit hit → Close >= Entry + ATR * tp_mult
        - Trend breaks → Close < SMA20 or RSI < 45

    Returns:
        dict {
            "df": DataFrame with equity curve
            "total_return": float
            "max_drawdown": float
        }
    """

    if df is None or df.empty:
        return None

    # Sort data
    df = df.sort_values("Date").copy()

    # Basic Indicators
    df["SMA_20"] = df["Close"].rolling(20).mean()

    # RSI + ATR (with safe try/except)
    try:
        df["RSI_14"] = ta.momentum.rsi(df["Close"], window=14)
        df["ATR_14"] = ta.volatility.average_true_range(
            df["High"], df["Low"], df["Close"], window=14
        )
    except Exception:
        return None

    # Drop rows with missing values
    df = df.dropna(subset=["SMA_20", "RSI_14", "ATR_14"])
    if df.empty:
        return None

    # Backtest State Variables
    position = 0                  # 0 = no trade, 1 = long
    entry_price = 0.0
    stop_price = 0.0
    take_profit = 0.0
    equity = 1.0
    equity_curve = []

    # Simulation Loop
    for _, row in df.iterrows():
        price = row["Close"]
        atr = row["ATR_14"]

        # --- Entry Condition: Trend + RSI Confirmation ---
        if position == 0:
            if price > row["SMA_20"] and row["RSI_14"] > 50:
                position = 1
                entry_price = price
                stop_price = entry_price - atr_mult_stop * atr
                take_profit = entry_price + atr_mult_tp * atr

        # --- Exit Logic ---
        else:
            # Stop-loss
            if price <= stop_price:
                equity *= price / entry_price
                position = 0

            # Take-profit
            elif price >= take_profit:
                equity *= price / entry_price
                position = 0

            # Trend breaks
            elif price < row["SMA_20"] or row["RSI_14"] < 45:
                equity *= price / entry_price
                position = 0

        equity_curve.append(equity)

    # Build Backtest Output
    df_bt = df.iloc[: len(equity_curve)].copy()
    df_bt["Equity"] = equity_curve

    # Total return
    total_return = df_bt["Equity"].iloc[-1] - 1.0

    # Max Drawdown
    max_equity = np.maximum.accumulate(df_bt["Equity"])
    dd = (df_bt["Equity"] - max_equity) / max_equity
    max_dd = float(dd.min()) if len(dd) else 0.0

    return {
        "df": df_bt,
        "total_return": float(total_return),
        "max_drawdown": float(max_dd),
    }
