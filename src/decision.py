from src.predictor import predict_latest
from src.sentiment import get_news_sentiment, get_overall_sentiment


def get_final_decision():

    prediction = predict_latest()

    news = get_news_sentiment()
    news_sentiment, news_score = get_overall_sentiment(news)

    model_prediction = prediction["prediction"]

    if model_prediction == "UP":
        if news_sentiment == "Bearish":
            signal = "HOLD"
        else:
            signal = "BUY"
    else:
        if news_sentiment == "Bullish":
            signal = "HOLD"
        else:
            signal = "NO BUY / EXIT"

    return {
        "date": prediction["date"],
        "close": prediction["close"],
        "model_prediction": model_prediction,
        "down_probability": prediction["down_probability"],
        "up_probability": prediction["up_probability"],
        "news_sentiment": news_sentiment,
        "news_score": news_score,
        "final_signal": signal
    }


if __name__ == "__main__":

    result = get_final_decision()

    print("\n===== FINAL MARKET DECISION =====")
    print(f"Date:              {result['date']}")
    print(f"Close Price:       ${result['close']:.2f}")
    print(f"ML Prediction:     {result['model_prediction']}")
    print(f"Down Probability:  {result['down_probability']:.2%}")
    print(f"Up Probability:    {result['up_probability']:.2%}")
    print(f"News Sentiment:    {result['news_sentiment']}")
    print(f"News Score:        {result['news_score']:.3f}")
    print(f"Final Signal:      {result['final_signal']}")