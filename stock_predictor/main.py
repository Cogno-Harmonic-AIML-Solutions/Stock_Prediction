from stock_predictor.src.config import TICKER, HORIZONS
from stock_predictor.src.data_loader import load_and_prepare_stock_data
from stock_predictor.src.features import add_rolling_features
from stock_predictor.src.models import train_model
from stock_predictor.src.backtest import backtest  # if modularized earlier
from stock_predictor.src.model import train_model
from stock_predictor.src.model_baseline import train_baseline_model
from stock_predictor.src.vader_sentiment import vader_sentiment_analysis
from stock_predictor.src.transformer import transformer_sentiment
from stock_predictor.src.news_scraper import fetch_et_news
from stock_predictor.src.sentiment.news_sentiment import fetch_and_analyze_sentiment

# ------------------------------ MAIN ENTRY POINT ------------------------------ #
import warnings
warnings.filterwarnings("ignore")

from modules.data_loader import load_stock_data
from modules.feature_engineering import apply_rolling_features
from modules.sentiment_vader import get_vader_sentiment_scores
from modules.sentiment_scraper import fetch_recent_headlines
from modules.sentiment_plot import plot_sentiment_vs_price


predictors = ["Close", "Volume", "Open", "High", "Low"]
model, baseline_precision = train_baseline_model(df, predictors)

def analyze_text(text):
    print("Analyzing sentiment for:", text)
    
    # VADER sentiment
    vader_result = vader_sentiment_analysis([text])
    print("VADER Sentiment:", vader_result)

    # Transformer-based sentiment
    transformer_result = transformer_sentiment(text)
    print("Transformer Sentiment:", transformer_result)

if __name__ == "__main__":
    # Load data
    df = load_data(TICKER)
    df = add_features(df, HORIZONS)
    df = load_and_prepare_stock_data(plot=True)

    # Analyze sample sentiment
    sample_text = "The stock market is showing signs of strong growth this quarter!"
    analyze_text(sample_text)

    # Feature columns for model
    feature_cols = [col for col in df.columns if "Ratio" in col or "Trend" in col]
    # Add new engineered features
    tatamotors, new_predictors = add_rolling_features(tatamotors)
    predictors += new_predictors

    # Backtest
    results = backtest(df, lambda tr, pr: train_model(tr, pr, 200, 50), feature_cols)
    # Run rolling window backtest
    backtest_results = backtest(df, model, predictors)

    # Evaluation
    from sklearn.metrics import precision_score
    print("Precision:", precision_score(results["Actual"], results["Predicted"]))

# Get the predict function and model
predict_fn, model = train_model(tatamotors, new_predictors)

# Run backtest with improved model
predictions = backtest(tatamotors, model, new_predictors, predict_fn=predict_fn)

# Display results
print("\n✅ Improved Model Prediction Counts:")
print(predictions["Predictions"].value_counts())
print("🎯 Improved Model Precision Score:", precision_score(predictions["Target"], predictions["Predictions"]))
print("📊 Actual Target Distribution:")
print(predictions["Target"].value_counts(normalize=True))

# Plot
predictions.plot(title="Actual vs Predicted Movement with New Features - Tata Motors")
plt.grid(True)
plt.ylabel("Direction (0 = Down, 1 = Up)")
plt.show()

# Final predictions
print("\n✅ Final Predictions:")
print(predictions.tail(10))

print("\n📰 Recent News from Economic Times:")
news_items = fetch_et_news("Tata-Motors", limit=5)

for item in news_items:
    print(f"- {item['title']}\n  Link: {item['link']}")

# Fetch and analyze news sentiment for Tata Motors
print("\n📰 Fetching news from yfinance or fallback to Economic Times...")
df_sentiment = fetch_and_analyze_sentiment(
    ticker_symbol="TATAMOTORS.NS",
    topic="Tata-Motors",
    limit=5,
    export=True
)


def main():
    # 1. Load and prepare data
    df = load_and_prepare_stock_data(plot=True)

    # 2. Add features
    df, new_predictors = add_rolling_features(df)

    # 3. Train and evaluate baseline
    baseline_model, baseline_precision = train_baseline_model(df, predictors=["Close", "Volume", "Open", "High", "Low"])

    # 4. Train improved model
    model, predict_fn = train_model(df, new_predictors)
    predictions = backtest(df, model, new_predictors, predict_fn=predict_fn)

    # 5. Sentiment analysis (headline-level)
    headlines = fetch_recent_headlines("TATAMOTORS.NS")
    get_vader_sentiment_scores(headlines)
    transformer_sentiment_analysis(headlines)  # New function

    # 6. Plot sentiment vs price
    plot_sentiment_vs_price(df)

    # 7. Aspect-level analysis
    run_aspect_sentiment_analysis(headlines)

    # 8. Final summary
    print_summary(baseline_precision, predictions)

if __name__ == "__main__":
    main()


# ------------------------------ TRANSFORMER-BASED + ASPECT-LEVEL ------------------------------ #
# Load multilingual transformer sentiment model
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create a pipeline
sentiment_pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Ensure headlines is defined
headlines = headlines if 'headlines' in locals() else []

# # Sample news headlines
headlines = [
    "Tata Motors reports record profits for Q1",
    "New electric vehicle lineup unveiled by Tata Motors",
    "Tata Motors stock drops after global chip shortage warning",
    "Strong sales boost Tata Motors’ revenue",
    "Tata Motors invests in EV battery production"
]

# 📰 Fallback: scrape Economic Times if yfinance shows no news
if not headlines:
    print("No news via yfinance — scraping Economic Times instead.")
    try:
        url = "https://economictimes.indiatimes.com/topic/Tata-Motors"
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")

        # Find article titles under proper containers (ET often uses .eachStory or .content)
        articles = soup.select(".tabdata .eachStory")
        headlines = []

        for article in articles:
            title = article.find("h3")
            if title:
                headlines.append(title.get_text(strip=True))
            if len(headlines) >= 5:
                break

        if headlines:
            print(f"✅ Scraped {len(headlines)} valid headlines from Economic Times:")
            for h in headlines:
                print("→", h)
        else:
            print("⚠️ Still no valid headlines found. Check page structure or try a different source.")
    except Exception as e:
        print(f"❌ Scraping failed: {e}")
        headlines = []

# Proceed only if we have headlines
if headlines:
    # Define aspects to track
    aspects = ["profit", "sales", "EV", "electric vehicle", "revenue", "investment", "emission", "partnership", "loss"]

    # Analyze aspect-level sentiment
    aspect_sentiment_results = []
    for hl in headlines:
        for asp in aspects:
            if asp.lower() in hl.lower():
                sentiment = sentiment_pipe(hl)[0]
                aspect_sentiment_results.append({
                    "headline": hl,
                    "aspect": asp,
                    "label": sentiment['label'],
                    "score": sentiment['score']
                })

    # Convert to DataFrame and display
    aspect_df = pd.DataFrame(aspect_sentiment_results)
    print("\n🧠 Aspect-Level Sentiment Analysis Results:")
    print(aspect_df)

    # Export results
    aspect_df.to_csv("aspect_sentiment_results.csv", index=False)
    aspect_df.to_excel("aspect_sentiment_results.xlsx", index=False)
    print("📄 Exported to aspect_sentiment_results.csv and .xlsx")
else:
    print("⚠️ No headlines available for aspect-level sentiment analysis.")

# Quick Summary
print("\n📈 Model Summary:")
print("Baseline Features  → Precision:", round(precision, 3))
print("Engineered Features → Precision:", round(precision_score(predictions["Target"], predictions["Predictions"]), 3))


print("\n📈 Summary:")
print(f"Baseline Precision → {round(baseline_precision, 3)}")
print(f"Improved Model Precision → {round(precision_score(predictions['Target'], predictions['Predictions']), 3)}")
