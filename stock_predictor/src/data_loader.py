# stock_predictor/src/data_loader.py

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def load_and_prepare_stock_data(plot: bool = True) -> pd.DataFrame:
    # Display all columns
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    # Load Tata Motors data
    tatamotors = yf.Ticker("TATAMOTORS.NS").history(period="max")

    # Clean up
    tatamotors.drop(columns=["Dividends", "Stock Splits"], inplace=True)
    tatamotors.dropna(inplace=True)

    print("\n✅ Initial Data:")
    print(tatamotors.head())

    # Optional plot
    if plot:
        tatamotors.plot.line(y="Close", use_index=True)
        plt.title("Tata Motors Closing Price Over Time")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.grid(True)
        plt.show()

    # Add target column
    tatamotors["Target"] = (tatamotors["Close"].shift(-1) > tatamotors["Close"]).astype(int)
    print("\n✅ Data with Target:")
    print(tatamotors[["Close", "Target"]].tail())

    return tatamotors
