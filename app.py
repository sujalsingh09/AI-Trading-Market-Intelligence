import streamlit as st
import pandas as pd
from pathlib import Path

from src.predictor import predict_latest
from src.news import get_news

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "NVDA_clean.csv"


st.set_page_config(
    page_title="NVDA Market Intelligence",
    page_icon="📈",
    layout="wide"
)


st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    max-width: 1250px;
}

.title {
    font-size: 38px;
    font-weight: 700;
}

.subtitle {
    color: #777;
    margin-bottom: 30px;
}

.signal-box {
    padding: 24px;
    border-radius: 16px;
    border: 1px solid #ddd;
    text-align: center;
}

.news-card {
    padding: 16px;
    border: 1px solid #ddd;
    border-radius: 12px;
    margin-bottom: 12px;
}

.news-meta {
    color: #777;
    font-size: 14px;
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)


def load_data():

    if not DATA_FILE.exists():
        return pd.DataFrame()

    data = pd.read_csv(DATA_FILE)

    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(
            data["Date"],
            errors="coerce",
            utc=True
        )

        data = data.dropna(subset=["Date"])
        data = data.sort_values("Date")

    return data


def prepare_indicators(data):

    data = data.copy()

    if "Close" not in data.columns:
        return data

    close = data["Close"]

    data["Daily_Return"] = close.pct_change() * 100

    data["Volatility_20D"] = (
        data["Daily_Return"]
        .rolling(20)
        .std()
    )

    data["SMA_20"] = (
        close
        .rolling(20)
        .mean()
    )

    data["SMA_50"] = (
        close
        .rolling(50)
        .mean()
    )

    data["SMA_Trend"] = (
        data["SMA_20"] - data["SMA_50"]
    )

    data["EMA_20"] = (
        close
        .ewm(span=20, adjust=False)
        .mean()
    )

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    data["RSI_14"] = 100 - (
        100 / (1 + rs)
    )

    ema_12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema_26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    data["MACD"] = ema_12 - ema_26

    data["MACD_Signal"] = (
        data["MACD"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    data["MACD_Histogram"] = (
        data["MACD"] -
        data["MACD_Signal"]
    )

    if "Volume" in data.columns:

        data["Volume_SMA_20"] = (
            data["Volume"]
            .rolling(20)
            .mean()
        )

        data["Volume_Ratio"] = (
            data["Volume"] /
            data["Volume_SMA_20"]
        )

    return data


def get_sentiment(news):

    if not news:
        return "Neutral", 0.0

    scores = []

    for item in news:

        title = str(
            item.get("title", "")
        )

        if not title:
            continue

        if TextBlob:

            score = TextBlob(
                title
            ).sentiment.polarity

        else:

            positive = [
                "rally",
                "rise",
                "gain",
                "growth",
                "strong",
                "bullish",
                "surge",
                "beat"
            ]

            negative = [
                "fall",
                "drop",
                "decline",
                "negative",
                "bearish",
                "concern",
                "risk",
                "loss"
            ]

            text = title.lower()

            pos = sum(
                word in text
                for word in positive
            )

            neg = sum(
                word in text
                for word in negative
            )

            score = (
                pos - neg
            ) / max(pos + neg, 1)

        scores.append(score)

    if not scores:
        return "Neutral", 0.0

    score = sum(scores) / len(scores)

    if score > 0.05:
        sentiment = "Bullish"

    elif score < -0.05:
        sentiment = "Bearish"

    else:
        sentiment = "Neutral"

    return sentiment, score


def value(row, column):

    try:

        x = row[column]

        if pd.isna(x):
            return "N/A"

        return f"{float(x):.2f}"

    except:

        return "N/A"


try:
    prediction = predict_latest()
except Exception as e:

    st.error(
        f"Prediction error: {e}"
    )

    st.stop()


data = load_data()

data = prepare_indicators(data)


try:
    news = get_news()
except Exception:
    news = []


sentiment, news_score = get_sentiment(
    news
)


price = prediction.get(
    "close",
    0
)

direction = prediction.get(
    "prediction",
    "N/A"
)

up_probability = prediction.get(
    "up_probability",
    0
)

down_probability = prediction.get(
    "down_probability",
    0
)

signal = prediction.get(
    "trading_signal",
    "N/A"
)


st.markdown(
    '<div class="title">📈 NVDA Market Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine learning, technical indicators and financial news in one place.'
    '</div>',
    unsafe_allow_html=True
)


st.subheader("Market Snapshot")


c1, c2, c3, c4, c5 = st.columns(5)


with c1:
    st.metric(
        "NVDA Price",
        f"${price:,.2f}"
    )


with c2:
    st.metric(
        "ML Prediction",
        direction
    )


with c3:
    st.metric(
        "Up Probability",
        f"{up_probability:.2%}"
    )


with c4:
    st.metric(
        "Down Probability",
        f"{down_probability:.2%}"
    )


with c5:
    st.metric(
        "News Sentiment",
        sentiment
    )


st.write("")


st.subheader("🎯 AI Decision Engine")


if "BUY" in str(signal).upper():

    icon = "🟢"

else:

    icon = "🔴"


st.markdown(
    f"""
    <div class="signal-box">
        <div style="font-size:32px">{icon}</div>
        <div style="font-size:28px;font-weight:700">
            {signal}
        </div>
        <div style="color:#777">
            ML: {direction} &nbsp; • &nbsp;
            News: {sentiment}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


st.write("")


if not data.empty:

    st.subheader("📊 Market Overview")

    chart = data[["Date", "Close"]].dropna().copy()

    chart["Date"] = pd.to_datetime(
        chart["Date"],
        errors="coerce",
        utc=True
    ).dt.tz_localize(None)

    chart["Close"] = pd.to_numeric(
        chart["Close"],
        errors="coerce"
    )

    chart = (
        chart
        .dropna()
        .drop_duplicates("Date")
        .sort_values("Date")
    )

    st.line_chart(
        chart,
        x="Date",
        y="Close",
        height=350
    )

st.subheader("🤖 Model Probability")


p1, p2 = st.columns(2)


with p1:
    st.metric(
        "DOWN",
        f"{down_probability:.2%}"
    )


with p2:
    st.metric(
        "UP",
        f"{up_probability:.2%}"
    )


probability = pd.DataFrame(
    {
        "Probability": [
            down_probability,
            up_probability
        ]
    },
    index=[
        "DOWN",
        "UP"
    ]
)


st.bar_chart(
    probability,
    height=250
)


st.subheader("📌 Technical Indicators")

if not data.empty:

    latest = data.iloc[-1]

    indicators = [
        ("Close", "Close"),
        ("SMA 20", "SMA_20"),
        ("SMA 50", "SMA_50"),
        ("EMA 20", "EMA_20"),
        ("RSI 14", "RSI_14"),
        ("MACD", "MACD"),
        ("MACD Signal", "MACD_Signal"),
        ("Daily Return", "Daily_Return"),
        ("Volatility 20D", "Volatility_20D"),
        ("Volume Ratio", "Volume_Ratio")
    ]

    for start in range(0, len(indicators), 5):

        cols = st.columns(5)

        for col, (name, column) in zip(
            cols,
            indicators[start:start + 5]
        ):
            with col:

                # Safely get value
                if column in data.columns:
                    raw_value = latest[column]

                    if pd.notna(raw_value):
                        display_value = f"{float(raw_value):.2f}"
                    else:
                        display_value = "N/A"
                else:
                    display_value = "N/A"

                st.metric(
                    label=name,
                    value=display_value
                )
st.write("")


st.subheader("📰 Latest NVDA News")


if news:

    for item in news[:10]:

        title = item.get(
            "title",
            "Untitled"
        )

        publisher = item.get(
            "publisher",
            item.get(
                "source",
                "Unknown"
            )
        )

        published = item.get(
            "published",
            ""
        )

        link = item.get(
            "link",
            ""
        )

        st.markdown(
            f"""
            <div class="news-card">
                <b>{title}</b>
                <div class="news-meta">
                    {publisher} • {published}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if link:
            st.markdown(
                f"[Read article →]({link})"
            )

else:

    st.info(
        "No recent NVDA news available."
    )


st.write("")


st.subheader("Market Summary")


s1, s2, s3 = st.columns(3)


with s1:

    st.metric(
        "News Score",
        f"{news_score:.3f}"
    )


with s2:

    if not data.empty:

        last_date = data[
            "Date"
        ].iloc[-1]

        st.metric(
            "Last Market Data",
            last_date.strftime(
                "%d %b %Y"
            )
        )

    else:

        st.metric(
            "Last Market Data",
            "N/A"
        )


with s3:

    if not data.empty:

        st.metric(
            "20D Volatility",
            value(
                data.iloc[-1],
                "Volatility_20D"
            )
        )

    else:

        st.metric(
            "20D Volatility",
            "N/A"
        )


st.divider()

st.caption(
    "NVDA AI Trading Market Intelligence • "
    "Machine Learning + Technical Analysis + News Analysis"
)