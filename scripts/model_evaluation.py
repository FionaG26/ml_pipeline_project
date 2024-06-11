from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import joblib
import numpy as np

# Load the trained model from the saved file
best_model = joblib.load('../models/best_model.pkl')

# Load X_test from the saved file or from your data preprocessing
X_test = pd.read_pickle('../models/X_preprocessed.pkl') 

# Load y_test from the saved file or from your data preprocessing
y_test = pd.read_pickle('../models/y_test.pkl')  

# Convert X_test to DataFrame if it's a NumPy array
if isinstance(X_test, np.ndarray):
    X_test = pd.DataFrame(X_test)

# Predictions
y_pred = best_model.predict(X_test)

# Print shapes for debugging
print("Shape of y_test:", y_test.shape)
print("Shape of y_pred:", y_pred.shape)

# Print out some samples from X_test and y_test for inspection
print("Samples from X_test:")
print(X_test.head())

print("Samples from y_test:")
print(y_test.head())

# Evaluation
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))
