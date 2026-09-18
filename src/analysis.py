from src.predictor import predict_latest
from src.news import get_news


def analyze_market():
    prediction = predict_latest()
    news = get_news()

    print("\n===== NVDA MARKET INTELLIGENCE =====")
    print(f"Date: {prediction['date']}")
    print(f"Close Price: ${prediction['close']:.2f}")
    print(f"Prediction: {prediction['prediction']}")
    print(f"Down Probability: {prediction['down_probability']:.2%}")
    print(f"Up Probability: {prediction['up_probability']:.2%}")
    print(f"Trading Signal: {prediction['trading_signal']}")

    print("\n===== LATEST NEWS =====")

    if not news:
        print("No recent news found.")
    else:
        for item in news[:5]:
            print(f"\nTitle: {item['title']}")
            print(f"Source: {item['publisher']}")
            print(f"Date: {item['published']}")
            print(f"Link: {item['link']}")
            print("-" * 60)

    return {
        "prediction": prediction,
        "news": news
    }


if __name__ == "__main__":
    analyze_market()