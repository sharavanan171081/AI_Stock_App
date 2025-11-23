import pandas as pd
import plotly.graph_objects as go
import ta


# ============================================================
#   ADD TECHNICAL INDICATORS
# ============================================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds SMA, RSI, MACD indicators safely.
    Works for NIFTY-50 large datasets without errors.
    """

    df = df.sort_values("Date").copy()

    # Basic Moving Averages
    df["SMA_5"] = df["Close"].rolling(5).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()

    # RSI
    try:
        df["RSI_14"] = ta.momentum.rsi(df["Close"], window=14)
    except Exception:
        df["RSI_14"] = None

    # MACD + Signal
    try:
        df["MACD"] = ta.trend.macd(df["Close"], window_fast=12, window_slow=26)
        df["MACD_SIGNAL"] = ta.trend.macd_signal(
            df["Close"], window_fast=12, window_slow=26, window_sign=9
        )
    except Exception:
        df["MACD"] = None
        df["MACD_SIGNAL"] = None

    return df


# ============================================================
#   CANDLESTICK + SMAs
# ============================================================
def make_candlestick_with_sma(df: pd.DataFrame, symbol: str) -> go.Figure:
    """
    Professional candlestick chart with SMA overlays.
    """

    fig = go.Figure()

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        )
    )

    # Add Moving Averages
    ma_list = [
        ("SMA_5", "SMA 5"),
        ("SMA_20", "SMA 20"),
        ("SMA_50", "SMA 50"),
    ]

    for col, name in ma_list:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["Date"],
                    y=df[col],
                    mode="lines",
                    name=name,
                )
            )

    # Layout
    fig.update_layout(
        title=f"{symbol} – Candlestick with Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=550,
        margin=dict(l=10, r=10, t=40, b=40),
    )

    return fig


# ============================================================
#   RSI CHART
# ============================================================
def make_rsi_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """
    RSI panel for technical overview.
    """

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["RSI_14"], mode="lines", name="RSI 14")
    )

    # Add Overbought/Oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")

    fig.update_layout(
        title=f"{symbol} – RSI (14)",
        xaxis_title="Date",
        yaxis_title="RSI",
        yaxis=dict(range=[0, 100]),
        height=250,
        margin=dict(l=10, r=10, t=40, b=40),
    )

    return fig


# ============================================================
#   MACD CHART
# ============================================================
def make_macd_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """
    MACD panel including signal line.
    """

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["MACD"], mode="lines", name="MACD")
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"], y=df["MACD_SIGNAL"], mode="lines", name="Signal"
        )
    )

    fig.update_layout(
        title=f"{symbol} – MACD",
        xaxis_title="Date",
        yaxis_title="MACD",
        height=250,
        margin=dict(l=10, r=10, t=40, b=40),
    )

    return fig
