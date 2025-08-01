# main.py

import sys
import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings("ignore")

from stock_predictor.src.config import TICKER, HORIZONS
from stock_predictor.src.data_loader import load_and_prepare_stock_data
from stock_predictor.src.features import add_rolling_features
from stock_predictor.src.models import train_model
from stock_predictor.src.backtest import backtest
from stock_predictor.src.model_baseline import train_baseline_model
from stock_predictor.src.vader import vader_sentiment_analysis
from stock_predictor.src.transformer import transformer_sentiment
from stock_predictor.src.news_scraper import fetch_et_news
from stock_predictor.src.sentiment.news_sentiment import fetch_and_analyze_sentiment


def analyze_sample_text():
    sample_text = "The stock market is showing signs of strong growth this quarter!"
    print("🔍 Analyzing sentiment for:", sample_text)
    print("VADER:", vader_sentiment_analysis([sample_text]))
    print("Transformer:", transformer_sentiment(sample_text))


def run_aspect_sentiment_analysis(headlines):
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    sentiment_pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    aspects = ["profit", "sales", "EV", "electric vehicle", "revenue", "investment", "emission", "partnership", "loss"]
    aspect_results = []

    for hl in headlines:
        for asp in aspects:
            if asp.lower() in hl.lower():
                sentiment = sentiment_pipe(hl)[0]
                aspect_results.append({
                    "headline": hl,
                    "aspect": asp,
                    "label": sentiment['label'],
                    "score": sentiment['score']
                })

    df_aspect = pd.DataFrame(aspect_results)
    print("\n🧠 Aspect-Level Sentiment:")
    print(df_aspect)
    df_aspect.to_csv("aspect_sentiment_results.csv", index=False)
    df_aspect.to_excel("aspect_sentiment_results.xlsx", index=False)


def main():
    df = load_and_prepare_stock_data(ticker="TATAMOTORS.NS", plot=True)

    # Baseline model
    baseline_model, baseline_precision = train_baseline_model(df, predictors=["Close", "Volume", "Open", "High", "Low"])

    # Add rolling features
    df, feature_cols = add_rolling_features(df, horizons=HORIZONS)

    # Improved model
    predict_fn, model = train_model(df, feature_cols)
    # predictions = backtest(df, model, feature_cols, predict_fn=predict_fn)
    predictions = backtest(df, model, feature_cols)


    print("\n✅ Prediction Summary:")
    print(predictions["Predictions"].value_counts())
    print("🎯 Precision Score:", precision_score(predictions["Target"], predictions["Predictions"]))

    predictions.plot(title="Actual vs Predicted Movement - Tata Motors")
    plt.grid(True)
    plt.ylabel("Direction (0 = Down, 1 = Up)")
    plt.show()

    # Sentiment
    analyze_sample_text()

    print("\n📰 Recent Headlines from Economic Times:")
    headlines_data = fetch_et_news("Tata-Motors", limit=5)
    headlines = [item["title"] for item in headlines_data]
    for item in headlines_data:
        print(f"- {item['title']}\n  Link: {item['link']}")

    # News Sentiment
    fetch_and_analyze_sentiment(
        ticker_symbol="TATAMOTORS.NS",
        topic="Tata-Motors",
        limit=5,
        export=True
    )

    # Aspect-level
    run_aspect_sentiment_analysis(headlines)

    print("\n📈 Precision Summary:")
    print("Baseline:", round(baseline_precision, 3))
    print("Improved:", round(precision_score(predictions['Target'], predictions['Predictions']), 3))


if __name__ == "__main__":
    main()
