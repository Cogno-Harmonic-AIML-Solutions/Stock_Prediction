import nltk                                                                             # Natural Language Toolkit for text processing (not used in this script but can be useful for future enhancements)
from nltk.sentiment.vader import SentimentIntensityAnalyzer                             # For sentiment analysis (not used in this script but can be useful for future enhancements)
# from nltk.sentiment.pysentimiento import PySentimiento, SentimentIntensityAnalyzer as PySentimentIntensityAnalyzer  # For sentiment analysis (not used in this script but can be useful for future enhancements)
# from pysentimiento import sentiment_analysis as pysentiment_analysis  # For sentiment analysis (not used in this script but can be useful for future enhancements)

nltk.download("vader_lexicon")                                          # Downloading VADER lexicon for sentiment analysis (not used in this script but can be useful for future enhancements)
nltk.download("pysentimiento")                                           # Downloading PySentimiento for sentiment analysis (not used in this script but can be useful for future enhancements)
sia = SentimentIntensityAnalyzer()