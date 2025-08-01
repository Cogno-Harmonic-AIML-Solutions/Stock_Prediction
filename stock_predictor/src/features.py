# stock_predictor/src/features.py

import pandas as pd

def add_rolling_features(df, target_col="Target", close_col="Close", horizons=[2, 5, 20, 60, 250]):
    """
    Adds rolling close ratios and trend features to the dataframe.

    Parameters:
        df (pd.DataFrame): Input dataframe with historical stock data
        target_col (str): Column used to compute trend direction
        close_col (str): Column used to compute rolling ratios
        horizons (list): List of time horizons for rolling computations

    Returns:
        pd.DataFrame: Modified dataframe with new features
        list: List of newly created feature names
    """
    new_features = []

    for horizon in horizons:
        ratio_col = f"{close_col}_Ratio_{horizon}"
        trend_col = f"Trend_{horizon}"

        df[ratio_col] = df[close_col] / df[close_col].rolling(horizon).mean()
        df[trend_col] = df[target_col].shift(1).rolling(horizon).sum()

        new_features += [ratio_col, trend_col]

    return df, new_features
