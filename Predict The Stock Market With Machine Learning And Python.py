# Predict The Stock Market With Machine Learning And Python
# This script predicts the stock market movement of Tata Motors using machine learning techniques in Python.
# It includes data collection, feature engineering, model training, evaluation, and visualization of results.
# The script uses libraries such as yfinance for data retrieval, pandas for data manipulation, and scikit-learn for machine learning.
# The model is built using a Random Forest Classifier, and it includes steps for backtesting and feature engineering to improve prediction accuracy.
# The script also incorporates sentiment analysis of news articles related to Tata Motors to enhance the model's predictive capabilities.
# The final model is evaluated based on precision scores and visualized through plots comparing actual vs predicted stock movements.
# This script serves as a comprehensive guide for predicting stock market movements using machine learning and Python.
# pip install yfinance matplotlib pandas scikit-learn requests beautifulsoup4 nltk transformers torch

# ------------------------------ PREDICT THE STOCK MARKET WITH MACHINE LEARNING AND PYTHON ------------------------------ #
# Importing required libraries
import yfinance as yf                                                                   # Importing yfinance
import matplotlib.pyplot as plt                                                         # For plotting graphs
import pandas as pd                                                                     # For data manipulation and analysis
from sklearn.ensemble import RandomForestClassifier                                     # Machine learning model for classification
from sklearn.metrics import precision_score                                             # For evaluating model performance
import requests                                                                         # For making HTTP requests to fetch data
from bs4 import BeautifulSoup                                                           # For web scraping (not used in this script but can be useful for future enhancements)
import nltk                                                                             # Natural Language Toolkit for text processing (not used in this script but can be useful for future enhancements)
from nltk.sentiment.vader import SentimentIntensityAnalyzer                             # For sentiment analysis (not used in this script but can be useful for future enhancements)
from datetime import datetime, timedelta                                                # For handling date and time (not used in this script but can be useful for future enhancements)
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline    # For advanced NLP tasks (not used in this script but can be useful for future enhancements)
import torch                                                                            # For deep learning tasks (not used in this script but can be useful for future enhancements)
import warnings                                                                         # For handling warnings
warnings.simplefilter(action="ignore", category=FutureWarning)                          # Ignore warnings for cleaner output
headlines = []                                                                          # Define globally in case both sources fail
# import pkg_resources                                                                    # For checking package versions (not used in this script but can be useful for future enhancements)

# ------------------------------ INITIAL SETUP ------------------------------ #
# Downloading NLTK resources
nltk.download("vader_lexicon")                                          # Downloading VADER lexicon for sentiment analysis (not used in this script but can be useful for future enhancements)
# nltk.download("pysentimento")                                           # Downloading PySentimiento for sentiment analysis (not used in this script but can be useful for future enhancements)
sia = SentimentIntensityAnalyzer()

# ------------------------------ DATA SETUP ------------------------------ #
# Display full dataframe columns and width
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# Step 1: Download historical stock data of Tata Motors from Yahoo Finance
tatamotors = yf.Ticker("TATAMOTORS.NS")                                 # 'TATAMOTORS.NS' is the NSE symbol for Tata Motors
tatamotors = tatamotors.history(period="max")                           # Load maximum available historical data

# Step 2: Drop unnecessary columns
# print("\n✅ Initial Data:")
# print(tatamotors.head())
tatamotors.drop(columns=["Dividends", "Stock Splits"], inplace=True)    # Remove columns not useful for prediction
tatamotors.dropna(inplace=True)                                         # Remove rows with missing values
print("\n✅ Initial Data:")
print(tatamotors.head())

# Step 3: Plotting closing price over time
tatamotors.plot.line(y="Close", use_index=True)
plt.title("Tata Motors Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.grid(True)
plt.show()

# Step 4: Create Target Column: Predict if the next day's price will go up (1) or down (0)
tatamotors["Target"] = (tatamotors["Close"].shift(-1) > tatamotors["Close"]).astype(int)
print("\n✅ Data with Target:")
print(tatamotors[["Close", "Target"]].tail())

# ------------------------------ BASELINE MODEL ------------------------------ #

# Step 5: Prepare data for training
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
train = tatamotors.iloc[:-100]                                          # Training data (excluding last 100 rows)
test = tatamotors.iloc[-100:]                                           # Testing data (last 100 rows)
predictors = ["Close", "Volume", "Open", "High", "Low"]

# Step 6: Train the baseline model
model.fit(train[predictors], train["Target"])

# Step 7: Make predictions and evaluate
preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)
precision = precision_score(test["Target"], preds)
print("\n🎯 Baseline Precision Score:", precision)

# Step 8: Plot actual vs predicted movement
test_results = pd.concat([test["Target"], preds], axis=1)
test_results.columns = ["Actual", "Predicted"]
test_results.plot(title="Actual vs Predicted Movement - Tata Motors")
plt.grid(True)
plt.ylabel("Direction (0 = Down, 1 = Up)")
plt.show()

# ------------------------------ BACKTESTING SYSTEM ------------------------------ #

# Function to train and test on a rolling window of data
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    return pd.concat([test["Target"], preds], axis=1)

def backtest(data, model, predictors, start=500, step=100):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

# Step 9: Run backtest with baseline predictors
predictions = backtest(tatamotors, model, predictors)
print("\n🔢 Prediction Distribution:")
print(predictions["Predictions"].value_counts())
print("🎯 Precision Score:", precision_score(predictions["Target"], predictions["Predictions"]))
print("📊 Actual Target Distribution:")
print(predictions["Target"].value_counts(normalize=True))

# ------------------------------ FEATURE ENGINEERING ------------------------------ #

# Step 10: Add more informative features (ratios and trend)
horizons = [2, 5, 20, 60, 250]   # Different time horizons for trend and moving average analysis
new_predictors = []

for horizon in horizons:
    rolling_averages = tatamotors.rolling(horizon).mean()
    ratio_column = f"Close_Ratio_{horizon}"
    tatamotors[ratio_column] = tatamotors["Close"] / rolling_averages["Close"]

    trend_column = f"Trend_{horizon}"
    tatamotors[trend_column] = tatamotors["Target"].shift(1).rolling(horizon).sum()

    new_predictors += [ratio_column, trend_column]

# ------------------------------ IMPROVED MODEL ------------------------------ #

# Step 11: Use probability thresholds for prediction
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

# Modified predict function using probability threshold
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    probs = model.predict_proba(test[predictors])[:, 1]  # Probability of class 1 (price increase)
    preds = (probs >= 0.6).astype(int)                   # Apply custom threshold
    preds = pd.Series(preds, index=test.index, name="Predictions")
    return pd.concat([test["Target"], preds], axis=1)

# Step 12: Run backtest with engineered features
predictions = backtest(tatamotors, model, new_predictors)
print("\n✅ Improved Model Prediction Counts:")
print(predictions["Predictions"].value_counts())
print("🎯 Improved Model Precision Score:", precision_score(predictions["Target"], predictions["Predictions"]))

print("📊 Actual Target Distribution:")
print(predictions["Target"].value_counts(normalize=True))
# Step 13: Plot actual vs predicted movement with new features
predictions.plot(title="Actual vs Predicted Movement with New Features - Tata Motors")
plt.grid(True)
plt.ylabel("Direction (0 = Down, 1 = Up)")
plt.show()
# Step 14: Display the final predictions
print("\n✅ Final Predictions:")
print(predictions.tail(10))

# ------------------------------ ADDING NEW ARTICLES ------------------------------ #
print("\n📰 Recent News from Economic Times:")
from bs4 import BeautifulSoup
url = "https://economictimes.indiatimes.com/topic/Tata-Motors"
res = requests.get(url)
soup = BeautifulSoup(res.text, "html.parser")
for headline in soup.select(".eachStory h3")[:5]:
    title = headline.get_text(strip=True)
    link = "https://economictimes.indiatimes.com" + headline.find_parent("a")["href"]
    print(f"- {title}")
    print(f"  Link: {link}\n")

# ------------------------------ ADDING SENTIMENT ANALYSIS ------------------------------ #
# Fetch Latest News
print("\n📰 Fetching latest Tata Motors headlines...")
# Use yfinance news items if available
ticker = yf.Ticker("TATAMOTORS.NS")                      # Keep this for accessing metadata like news
tatamotors = ticker.history(period="max")                # Use this for historical price data

# Later when fetching news:
try:
    news_items = ticker.news                             # Not tatamotors.news
    headlines = [item['title'] for item in news_items[:5] if 'title' in item]
except Exception as e:
    print("⚠️ Could not fetch news from yfinance:", e)
    headlines = []

# Fallback: scrape Economic Times if yfinance shows no news
if not headlines:
    print("No news via yfinance—scraping Economic Times instead.")
    from bs4 import BeautifulSoup
    res = requests.get("https://economictimes.indiatimes.com/topic/Tata-Motors")
    soup = BeautifulSoup(res.text, "html.parser")
    headlines = [h.get_text(strip=True) for h in soup.select(".eachStory h3")[:5]]

# Display headlines
for i, hl in enumerate(headlines, 1):
    print(f"{i}. {hl}")

# Apply Sentiment Analysis with VADER
analyzer = SentimentIntensityAnalyzer()
print("\n🧠 Sentiment Scores:")
for hl in headlines:
    scores = analyzer.polarity_scores(hl)
    compound = scores['compound']
    sentiment = "👍 Positive" if compound > 0.05 else "👎 Negative" if compound < -0.05 else "😐 Neutral"
    print(f"{hl}\n   Sentiment: {sentiment} (Compound score: {compound})\n")

# Save Results (Optional)
df_news = pd.DataFrame({
    "headline": headlines,
    "compound_score": [analyzer.polarity_scores(h)['compound'] for h in headlines]
})
df_news.to_csv("tatamotors_news_sentiment.csv", index=False)
print("News sentiment exported to tatamotors_news_sentiment.csv")

# ------------------------------ PLOTTING SENTIMENT ANALYSIS ------------------------------ #

def fetch_news_articles(company="Tata Motors"):
    to_date = datetime.today().strftime('%Y-%m-%d')
    from_date = (datetime.today() - timedelta(days=14)).strftime('%Y-%m-%d')  # Last 14 days only
    print(f"Fetching news from {from_date} to {to_date}")
    api_key = "b84a50ebd0f043e98213ed7badf373cb"  # NewsAPI key
    url = (
        f"https://newsapi.org/v2/everything?q={company}&from={from_date}&to={to_date}"
        f"&sortBy=publishedAt&language=en&pageSize=100&apiKey={api_key}"
    )
    response = requests.get(url)
    if response.status_code != 200:
        print(f"❌ Failed to fetch articles: {response.status_code} - {response.text}")
        return pd.DataFrame()

    articles = response.json().get("articles", [])
    if not articles:
        print("⚠️ No articles received from NewsAPI.")
        return pd.DataFrame()

    news = [
        {"date": article["publishedAt"][:10], "content": article["title"] + " " + (article["description"] or "")}
        for article in articles
    ]
    return pd.DataFrame(news)

news_df = fetch_news_articles()

if news_df.empty or "date" not in news_df.columns:
    print("❌ News data is empty or missing 'date' column. Skipping sentiment vs price plot.")
else:
    news_df["date"] = pd.to_datetime(news_df["date"])
    news_df["sentiment"] = news_df["content"].apply(lambda text: sia.polarity_scores(text)["compound"])

    # Group sentiment by day
    daily_sentiment = news_df.groupby(news_df["date"].dt.date)["sentiment"].mean()

    tatamotors = tatamotors.reset_index()  # Reset index to avoid conflict
    tatamotors["Date"] = tatamotors["Date"].dt.date  # Ensure 'Date' is in date format

    # Merge sentiment with stock data
    sentiment_merge = tatamotors.merge(daily_sentiment, left_on="Date", right_index=True)
    sentiment_merge.rename(columns={"sentiment": "News_Sentiment"}, inplace=True)

    # Plot Close Price vs Sentiment
    plt.figure(figsize=(14,6))
    plt.plot(sentiment_merge["Date"], sentiment_merge["Close"], label="Closing Price", color="blue")
    plt.plot(sentiment_merge["Date"], sentiment_merge["News_Sentiment"]*1000, label="News Sentiment (scaled)", color="red", linestyle='--')
    plt.title("Tata Motors: Stock Price vs News Sentiment")
    plt.xlabel("Date")
    plt.ylabel("Price / Sentiment")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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
