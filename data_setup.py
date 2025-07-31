from initial_setup import *                                                             # Importing everything from initial_setup for convenience
import yfinance as yf                                                                   # Importing yfinance again, redundant but included for clarity
import matplotlib.pyplot as plt                                                         # For plotting graphs
import pandas as pd                                                                     # For data manipulation and analysis

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