from vader import vader_sentiment
from transformer import transformer_sentiment

text = "Tata Motors is showing a bullish trend with increased investor confidence."

vader_score = vader_sentiment(text)
transformer_result = transformer_sentiment(text)

print(f"VADER Score: {vader_score}")
print(f"Transformer: {transformer_result['label']} ({transformer_result['score']:.2f})")
