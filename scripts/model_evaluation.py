from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import joblib

# Load the trained model from the saved file
best_model = joblib.load('../models/best_model.pkl')  # Assuming the best model is saved as 'best_model.pkl' in the models directory

# Load X_test from the saved file or from your data preprocessing
X_test = pd.read_pickle('../models/X_preprocessed.pkl')  # Assuming X_preprocessed.pkl is located in the models directory

# Load y_test from the saved file or from your data preprocessing
y_test = pd.read_pickle('../models/y_test.pkl')  # Assuming y_test.pkl is located in the models directory

# Predictions
y_pred = best_model.predict(X_test)

# Print shapes for debugging
print("Shape of y_test:", y_test.shape)
print("Shape of y_pred:", y_pred.shape)

# Evaluation
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))
