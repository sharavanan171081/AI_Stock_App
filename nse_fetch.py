import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

# ============================================================
#  CONFIG
# ============================================================
SYMBOLS = [
    "TCS", "HDFCBANK", "INFY", "RELIANCE", "ICICIBANK", "SBIN",
    "AXISBANK", "KOTAKBANK", "LT", "ITC", "HINDUNILVR", "BAJFINANCE",
    "ASIANPAINT", "MARUTI", "SUNPHARMA", "TECHM", "ULTRACEMCO",
    "BHARTIARTL", "POWERGRID", "NESTLEIND"
]

# Full historical data (Yahoo's maximum)
HISTORY_MODE = "max"

# ============================================================
#  AUTO-DETECT PROJECT ROOT (WORKS ON STREAMLIT CLOUD)
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_FILE = DATA_DIR / "stock_data.csv"


# ============================================================
#  FETCH SINGLE SYMBOL
# ============================================================
def fetch_symbol(symbol: str) -> pd.DataFrame | None:
    """Fetch OHLCV data for one NSE symbol using yfinance."""

    yf_symbol = symbol + ".NS"

    df = yf.download(
        yf_symbol,
        period=HISTORY_MODE,
        interval="1d",
        progress=False,
        auto_adjust=False,
    )

    if df.empty:
        print(f"⚠ No data for {symbol}")
        return None

    df = df.reset_index()

    # Flatten multi-index columns from Yahoo
    df.columns = [c if not isinstance(c, tuple) else c[0] for c in df.columns]

    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

    # Numeric clean
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

    df["Symbol"] = symbol

    return df


# ============================================================
#  MAIN CONTROLLER
# ============================================================
def main() -> None:

    DATA_DIR.mkdir(exist_ok=True)

    print(f"⏳ Fetching NSE data ({HISTORY_MODE}) for: {', '.join(SYMBOLS)}")

    frames: list[pd.DataFrame] = []

    for sym in SYMBOLS:
        print(f"→ Fetching {sym}...")
        df_sym = fetch_symbol(sym)
        if df_sym is not None and not df_sym.empty:
            frames.append(df_sym)

    if not frames:
        print("❌ No data fetched for any symbol.")
        return

    # Combine all symbols
    full_df = pd.concat(frames, ignore_index=True)

    # Sort by symbol + date
    full_df = full_df.sort_values(["Symbol", "Date"])

    # Convert date format (DD-MM-YYYY)
    full_df["Date"] = pd.to_datetime(full_df["Date"]).dt.strftime("%d-%m-%Y")

    # Columns in correct order
    full_df = full_df[["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]]

    # Save to CSV
    full_df.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ Saved ML-ready stock data to: {OUTPUT_FILE}")
    print(full_df.tail())


# ============================================================
#  ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
