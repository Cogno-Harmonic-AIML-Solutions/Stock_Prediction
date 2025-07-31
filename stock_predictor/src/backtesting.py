# stock_predictor/src/backtesting.py

import pandas as pd
from sklearn.metrics import precision_score

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    return pd.concat([test["Target"], preds], axis=1)

def backtest(data, model, predictors, start=500, step=100):
    print("\n🔁 Running rolling backtest...")
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    results = pd.concat(all_predictions)

    print("\n🔢 Prediction Distribution:")
    print(results["Predictions"].value_counts())
    print("🎯 Precision Score:", precision_score(results["Target"], results["Predictions"]))
    print("📊 Actual Target Distribution:")
    print(results["Target"].value_counts(normalize=True))

    return results
