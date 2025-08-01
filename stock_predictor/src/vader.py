# vader.py
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure lexicon is available
nltk.download("vader_lexicon")

def vader_sentiment_analysis(texts):
    sia = SentimentIntensityAnalyzer()
    results = []
    for text in texts:
        score = sia.polarity_scores(text)["compound"]
        results.append(score)
    return results
