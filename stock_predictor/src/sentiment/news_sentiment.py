import yfinance as yf
import requests
from bs4 import BeautifulSoup
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def fetch_headlines_yf(ticker_symbol="TATAMOTORS.NS", limit=5):
    """
    Try fetching news from Yahoo Finance.
    Returns a list of headlines.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        news_items = ticker.news
        headlines = [item['title'] for item in news_items[:limit] if 'title' in item]
        return headlines
    except Exception as e:
        print("⚠️ Could not fetch news from yfinance:", e)
        return []

def fallback_scrape_et(topic="Tata-Motors", limit=5):
    """
    Scrape Economic Times headlines if yfinance fails.
    """
    print("No news via yfinance—scraping Economic Times instead.")
    res = requests.get(f"https://economictimes.indiatimes.com/topic/{topic}")
    soup = BeautifulSoup(res.text, "html.parser")
    headlines = [h.get_text(strip=True) for h in soup.select(".eachStory h3")[:limit]]
    return headlines

def analyze_sentiment(headlines):
    """
    Apply VADER sentiment analysis to headlines.
    """
    analyzer = SentimentIntensityAnalyzer()
    results = []
    for hl in headlines:
        score = analyzer.polarity_scores(hl)['compound']
        sentiment = "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"
        results.append({"headline": hl, "compound_score": score, "sentiment": sentiment})
    return pd.DataFrame(results)

def fetch_and_analyze_sentiment(ticker_symbol="TATAMOTORS.NS", topic="Tata-Motors", limit=5, export=False):
    """
    Full pipeline: fetch headlines, fallback if needed, analyze sentiment, and optionally export.
    """
    headlines = fetch_headlines_yf(ticker_symbol, limit)
    if not headlines:
        headlines = fallback_scrape_et(topic, limit)

    print("\n🧠 Sentiment Scores:")
    df = analyze_sentiment(headlines)

    for _, row in df.iterrows():
        symbol = "👍" if row["sentiment"] == "Positive" else "👎" if row["sentiment"] == "Negative" else "😐"
        print(f"{row['headline']}\n   Sentiment: {symbol} {row['sentiment']} (Compound: {row['compound_score']})\n")

    if export:
        df.to_csv(f"{topic.lower().replace('-', '_')}_news_sentiment.csv", index=False)
        print(f"✅ News sentiment exported to {topic.lower().replace('-', '_')}_news_sentiment.csv")

    return df
