import pandas as pd
import numpy as np

from src.features import calculate_features


def create_sample_data(rows=100):

    dates = pd.date_range(
        start="2025-01-01",
        periods=rows,
        freq="D"
    )

    data = pd.DataFrame({
        "Open": np.random.uniform(100, 110, rows),
        "High": np.random.uniform(110, 120, rows),
        "Low": np.random.uniform(90, 100, rows),
        "Close": np.random.uniform(100, 110, rows),
        "Volume": np.random.randint(
            1_000_000,
            5_000_000,
            rows
        ),
        "Dividends": 0,
        "Stock Splits": 0
    }, index=dates)

    return data


def test_feature_generation():

    data = create_sample_data()

    result = calculate_features(data)

    required_features = [
        "Daily_Return",
        "Volatility_20D",
        "SMA_20",
        "SMA_50",
        "SMA_Trend",
        "EMA_20",
        "RSI_14",
        "MACD",
        "MACD_Signal",
        "MACD_Histogram",
        "Volume_SMA_20",
        "Volume_Ratio"
    ]

    for feature in required_features:
        assert feature in result.columns


def test_no_infinite_values():

    data = create_sample_data()

    result = calculate_features(data)

    numeric_data = result.select_dtypes(
        include="number"
    )

    assert not np.isinf(
        numeric_data.to_numpy()
    ).any()
