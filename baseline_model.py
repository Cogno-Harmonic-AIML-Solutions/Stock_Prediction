from data_setup import *                                                                # Importing data setup for historical stock data
import matplotlib.pyplot as plt                                                         # For plotting graphs
from sklearn.ensemble import RandomForestClassifier                                     # Machine learning model for classification
from sklearn.metrics import precision_score                                             # For evaluating model performance
import pandas as pd                                                                     # For data manipulation and analysis

# Step 5: Prepare data for training
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
train = data_setup.tatamotors.iloc[:-100]                                               # Training data (excluding last 100 rows)
test = data_setup.tatamotors.iloc[-100:]                                                # Testing data (last 100 rows)
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