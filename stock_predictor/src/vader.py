from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def vader_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)['compound']

if __name__ == "__main__":
    sample_text = "The market sentiment is extremely positive!"
    score = vader_sentiment(sample_text)
    print(f"VADER Sentiment Score: {score}")
