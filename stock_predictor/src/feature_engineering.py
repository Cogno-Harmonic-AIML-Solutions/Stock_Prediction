def add_features(df, horizons):
    for horizon in horizons:
        df[f"Close_Ratio_{horizon}"] = df["Close"] / df["Close"].rolling(horizon).mean()
        df[f"Trend_{horizon}"] = df["Target"].shift(1).rolling(horizon).sum()
    return df
