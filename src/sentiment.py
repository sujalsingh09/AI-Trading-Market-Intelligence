from textblob import TextBlob
from src.news import get_news


def analyze_sentiment(text):
    score = TextBlob(text).sentiment.polarity

    if score > 0.1:
        sentiment = "Positive"
    elif score < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment, score


def get_news_sentiment():
    news = get_news()
    results = []

    for item in news:
        sentiment, score = analyze_sentiment(item["title"])

        results.append({
            "title": item["title"],
            "publisher": item["publisher"],
            "published": item["published"],
            "link": item["link"],
            "sentiment": sentiment,
            "score": score
        })

    return results


def get_overall_sentiment(results):
    if not results:
        return "Neutral", 0.0

    scores = [item["score"] for item in results]
    average_score = sum(scores) / len(scores)

    if average_score > 0.1:
        sentiment = "Bullish"
    elif average_score < -0.1:
        sentiment = "Bearish"
    else:
        sentiment = "Neutral"

    return sentiment, average_score


if __name__ == "__main__":
    results = get_news_sentiment()

    print("\n===== NVDA NEWS SENTIMENT =====\n")

    for item in results:
        print(f"Title: {item['title']}")
        print(f"Sentiment: {item['sentiment']}")
        print(f"Score: {item['score']:.3f}")
        print("-" * 60)

    overall_sentiment, overall_score = get_overall_sentiment(results)

    print("\n===== OVERALL NEWS SENTIMENT =====")
    print(f"Sentiment: {overall_sentiment}")
    print(f"Score: {overall_score:.3f}")