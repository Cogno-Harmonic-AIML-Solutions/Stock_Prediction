from stock_predictor.src.config import TICKER, HORIZONS
from stock_predictor.src.data_loader import load_data
from stock_predictor.src.feature_engineering import add_features
from stock_predictor.src.backtest import backtest
from stock_predictor.src.model import train_model
from stock_predictor.src.vader import vader_sentiment
from stock_predictor.src.transformer import transformer_sentiment

def analyze_text(text):
    print("Analyzing sentiment for:", text)
    
    # VADER sentiment
    vader_result = vader_sentiment(text)
    print("VADER Sentiment:", vader_result)

    # Transformer-based sentiment
    transformer_result = transformer_sentiment(text)
    print("Transformer Sentiment:", transformer_result)

if __name__ == "__main__":
    # Load data
    df = load_data(TICKER)
    df = add_features(df, HORIZONS)

    # Analyze sample sentiment
    sample_text = "The stock market is showing signs of strong growth this quarter!"
    analyze_text(sample_text)

    # Feature columns for model
    feature_cols = [col for col in df.columns if "Ratio" in col or "Trend" in col]

    # Backtest
    results = backtest(df, lambda tr, pr: train_model(tr, pr, 200, 50), feature_cols)

    # Evaluation
    from sklearn.metrics import precision_score
    print("Precision:", precision_score(results["Actual"], results["Predicted"]))
