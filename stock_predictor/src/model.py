from sklearn.ensemble import RandomForestClassifier

def train_model(train, predictors, n_estimators=100, min_samples_split=100):
    model = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, random_state=1)
    model.fit(train[predictors], train["Target"])
    return model

def predict_model(model, test, predictors, threshold=0.6):
    probs = model.predict_proba(test[predictors])[:, 1]
    return (probs >= threshold).astype(int)
