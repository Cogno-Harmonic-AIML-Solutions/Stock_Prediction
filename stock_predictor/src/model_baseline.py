# stock_predictor/src/model_baseline.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

def train_baseline_model(df: pd.DataFrame, predictors: list, test_size: int = 100):
    print("\n🚀 Training baseline Random Forest model...")

    model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
    
    train = df.iloc[:-test_size]
    test = df.iloc[-test_size:]

    model.fit(train[predictors], train["Target"])
    
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index)
    
    precision = precision_score(test["Target"], preds)
    print("\n🎯 Baseline Precision Score:", precision)

    # Plotting results
    test_results = pd.concat([test["Target"], preds], axis=1)
    test_results.columns = ["Actual", "Predicted"]
    test_results.plot(title="Actual vs Predicted Movement - Tata Motors")
    plt.grid(True)
    plt.ylabel("Direction (0 = Down, 1 = Up)")
    plt.show()

    return model, precision
