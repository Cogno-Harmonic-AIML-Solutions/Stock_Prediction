import pandas as pd
from .model import predict_model

def backtest(data, model_fn, predictors, start=500, step=100, threshold=0.6):
    results = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[:i]
        test = data.iloc[i:i+step]
        model = model_fn(train, predictors)
        preds = predict_model(model, test, predictors, threshold)
        combined = pd.DataFrame({"Actual": test["Target"], "Predicted": preds}, index=test.index)
        results.append(combined)
    return pd.concat(results)
