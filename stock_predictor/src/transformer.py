from transformers import pipeline

def transformer_sentiment(text):
    sentiment_pipeline = pipeline("sentiment-analysis")
    result = sentiment_pipeline(text)[0]
    return {
        "label": result['label'],
        "score": result['score']
    }

if __name__ == "__main__":
    sample_text = "The stock crashed after weak quarterly results."
    result = transformer_sentiment(sample_text)
    print(f"Transformer Sentiment: {result['label']} ({result['score']:.2f})")
