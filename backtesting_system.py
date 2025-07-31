from baseline_model import *

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