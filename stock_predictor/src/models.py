# stock_predictor/src/models.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_model(train, predictors, threshold=0.6, random_state=1, n_estimators=200, min_samples_split=50):
    """
    Trains a RandomForest model and returns a prediction function using thresholding.

    Parameters:
        train (pd.DataFrame): Training dataset
        predictors (list): List of feature columns
        threshold (float): Probability threshold for classification
        random_state (int): Random state for reproducibility
        n_estimators (int): Number of trees in the forest
        min_samples_split (int): Minimum samples per split

    Returns:
        function: Predict function to be used for backtesting
        model: Trained model instance
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        random_state=random_state
    )

    def predict(train, test, predictors):
        model.fit(train[predictors], train["Target"])
        probs = model.predict_proba(test[predictors])[:, 1]
        preds = (probs >= threshold).astype(int)
        preds = pd.Series(preds, index=test.index, name="Predictions")
        return pd.concat([test["Target"], preds], axis=1)

    return predict, model
