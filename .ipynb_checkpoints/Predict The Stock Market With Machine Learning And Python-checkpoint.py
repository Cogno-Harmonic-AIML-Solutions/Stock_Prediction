# Importing required libraries
import yfinance as yf                                                   # Yahoo Finance for historical stock data
import matplotlib.pyplot as plt                                         # For plotting graphs
import pandas as pd                                                     # For data handling and analysis
from sklearn.ensemble import RandomForestClassifier                     # Machine Learning model - Random Forest
from sklearn.metrics import precision_score                             # Evaluation metric

# ------------------------------ DATA SETUP ------------------------------ #
# Display full dataframe columns and width
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# Step 1: Download historical stock data of Tata Motors from Yahoo Finance
tatamotors = yf.Ticker("TATAMOTORS.NS")                                 # 'TATAMOTORS.NS' is the NSE symbol for Tata Motors
tatamotors = tatamotors.history(period="max")                           # Load maximum available historical data

# Step 2: Drop unnecessary columns
print("\n✅ Initial Data:")
print(tatamotors.head())
tatamotors.drop(columns=["Dividends", "Stock Splits"], inplace=True)    # Remove columns not useful for prediction
tatamotors.dropna(inplace=True)                                         # Remove rows with missing values

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

# ------------------------------ SUMMARY AND NEXT STEPS WITH THE MODEL ------------------------------ #
