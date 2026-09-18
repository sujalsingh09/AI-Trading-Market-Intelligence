import requests
from datetime import datetime


def get_news():
    url = "https://query1.finance.yahoo.com/v1/finance/search"

    params = {
        "q": "NVDA",
        "newsCount": 20
    }

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=15
    )

    if response.status_code == 429:
        print("Yahoo Finance rate limit reached.")
        return []

    response.raise_for_status()

    data = response.json()
    news = []

    keywords = [
        "nvidia",
        "nvda",
        "geforce",
        "cuda",
        "blackwell",
        "rubin"
    ]

    for item in data.get("news", []):
        title = item.get("title", "")
        title_lower = title.lower()

        if not any(keyword in title_lower for keyword in keywords):
            continue

        timestamp = item.get("providerPublishTime")

        published = None

        if timestamp:
            published = datetime.fromtimestamp(timestamp)

        news.append({
            "title": title,
            "publisher": item.get("publisher", ""),
            "link": item.get("link", ""),
            "published": published
        })

    return news


if __name__ == "__main__":
    news = get_news()

    print("\n===== NVDA NEWS =====\n")

    if not news:
        print("No NVDA news found.")
    else:
        for item in news:
            print(f"Title: {item['title']}")
            print(f"Source: {item['publisher']}")
            print(f"Date: {item['published']}")
            print(f"Link: {item['link']}")
            print("-" * 70)