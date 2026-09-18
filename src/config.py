from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# DATA
# ============================================================

DATA_DIR = BASE_DIR / "data"

# Your actual file is CSV
DATA_FILE = DATA_DIR / "NVDA_clean.csv"


# ============================================================
# MODELS
# ============================================================

NOTEBOOK_DIR = BASE_DIR / "notebooks"

MODEL_FILE = NOTEBOOK_DIR / "nvda_final_model.pkl"

FEATURE_FILE = NOTEBOOK_DIR / "nvda_final_features.pkl"


# ============================================================
# ML FEATURES
# ============================================================

FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
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
    "Volume_Ratio",
]