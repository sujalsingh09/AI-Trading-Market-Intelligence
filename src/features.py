import pandas as pd
import numpy as np


def calculate_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate technical indicators used by the NVDA ML model.
    """

    df = data.copy()

    # Remove unnecessary columns if present
    for col in ["Dividends", "Stock Splits"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Daily return
    df["Daily_Return"] = df["Close"].pct_change()

    # 20-day volatility
    df["Volatility_20D"] = (
        df["Daily_Return"]
        .rolling(window=20)
        .std()
    )

    # Moving averages
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()

    # SMA trend
    df["SMA_Trend"] = df["SMA_20"] - df["SMA_50"]

    # EMA
    df["EMA_20"] = df["Close"].ewm(
        span=20,
        adjust=False
    ).mean()

    # RSI
    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema_26 = df["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = ema_12 - ema_26

    df["MACD_Signal"] = df["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["MACD_Histogram"] = (
        df["MACD"] - df["MACD_Signal"]
    )

    # Volume indicators
    df["Volume_SMA_20"] = (
        df["Volume"].rolling(20).mean()
    )

    df["Volume_Ratio"] = (
        df["Volume"] / df["Volume_SMA_20"]
    )

    return df


def prepare_latest_features(data: pd.DataFrame, features: list):
    """
    Generate features and return the latest valid row.
    """

    df = calculate_features(data)

    missing = [
        feature for feature in features
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )

    latest = (
        df[features]
        .dropna()
        .iloc[[-1]]
    )

    return latest
