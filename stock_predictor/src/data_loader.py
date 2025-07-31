import yfinance as yf
import pandas as pd

def load_data(ticker: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period="max")
    df.drop(columns=["Dividends", "Stock Splits"], inplace=True)
    df.dropna(inplace=True)
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df
