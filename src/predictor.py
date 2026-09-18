import joblib
import pandas as pd

from src.config import (
    DATA_FILE,
    MODEL_FILE,
    FEATURE_FILE,
)

from src.features import prepare_latest_features


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """
    Load the final trained Random Forest model.
    """
    return joblib.load(MODEL_FILE)


# ============================================================
# LOAD FEATURES
# ============================================================

def load_features():
    """
    Load the exact feature list used during model training.
    """
    return joblib.load(FEATURE_FILE)


# ============================================================
# LOAD MARKET DATA
# ============================================================

def load_market_data():
    """
    Load NVDA market data from CSV.

    Handles:
    - CSV format
    - Date column
    - Mixed timezone values
    - Missing dates
    """

    print(f"Loading data from: {DATA_FILE}")

    # Read CSV
    data = pd.read_csv(DATA_FILE)

    if data.empty:
        raise ValueError("NVDA_clean.csv is empty.")

    # --------------------------------------------------------
    # Find date column
    # --------------------------------------------------------

    possible_date_columns = [
        "Date",
        "date",
        "Datetime",
        "datetime",
        "Timestamp",
        "timestamp",
    ]

    date_column = None

    for col in possible_date_columns:
        if col in data.columns:
            date_column = col
            break

    if date_column is None:
        raise ValueError(
            f"No date column found. Available columns: {list(data.columns)}"
        )

    # --------------------------------------------------------
    # FIX MIXED TIMEZONES
    # --------------------------------------------------------

    data[date_column] = pd.to_datetime(
        data[date_column],
        errors="coerce",
        utc=True
    )

    # Remove invalid dates
    data = data.dropna(subset=[date_column])

    # Sort chronologically
    data = data.sort_values(date_column)

    # Set date as index
    data = data.set_index(date_column)

    # --------------------------------------------------------
    # Check required OHLCV columns
    # --------------------------------------------------------

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required market columns: {missing_columns}"
        )

    print(f"Loaded rows: {len(data)}")
    print(f"Latest date: {data.index[-1]}")
    print(f"Latest close: {data['Close'].iloc[-1]}")

    return data


# ============================================================
# PREDICT LATEST
# ============================================================

def predict_latest():

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Load feature list
    # --------------------------------------------------------

    features = load_features()

    print("\n===== LOADED MODEL =====")
    print(model)

    print("\n===== FEATURES =====")
    print(features)

    # --------------------------------------------------------
    # Load market data
    # --------------------------------------------------------

    data = load_market_data()

    # --------------------------------------------------------
    # Prepare latest features
    # --------------------------------------------------------

    latest_data = prepare_latest_features(
        data,
        features
    )

    if latest_data.empty:
        raise ValueError(
            "No valid row available after feature preparation."
        )

    # Make sure feature order is EXACTLY same as training
    latest_data = latest_data[features]

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(latest_data)[0]

    probabilities = model.predict_proba(
        latest_data
    )[0]

    down_probability = float(probabilities[0])
    up_probability = float(probabilities[1])

    # --------------------------------------------------------
    # Trading signal
    # --------------------------------------------------------

    if prediction == 1:
        direction = "UP"
        signal = "BUY"
    else:
        direction = "DOWN"
        signal = "NO BUY / EXIT"

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {
        "date": str(data.index[-1]),
        "close": float(data["Close"].iloc[-1]),
        "prediction": direction,
        "down_probability": down_probability,
        "up_probability": up_probability,
        "trading_signal": signal,
    }

    return result


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    result = predict_latest()

    print("\n")
    print("=" * 55)
    print("           FINAL NVDA PREDICTION")
    print("=" * 55)

    print(f"Date:              {result['date']}")
    print(f"Close Price:       ${result['close']:.2f}")

    print(
        f"Down Probability:  "
        f"{result['down_probability'] * 100:.2f}%"
    )

    print(
        f"Up Probability:    "
        f"{result['up_probability'] * 100:.2f}%"
    )

    print(f"Prediction:        {result['prediction']}")
    print(f"Trading Signal:    {result['trading_signal']}")

    print("=" * 55)