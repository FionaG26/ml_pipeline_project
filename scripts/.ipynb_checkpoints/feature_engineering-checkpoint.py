import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

# Load preprocessed data from file
X_preprocessed = joblib.load('../models/X_preprocessed.pkl')

# Load the original data to extract the target variable
data = pd.read_csv('../data/heart.csv')

# Define the target variable
y = data['output']

# Feature selection
selector = SelectKBest(score_func=f_classif, k='all')  # Adjust 'k' as needed
X_selected = selector.fit_transform(X_preprocessed, y)

# Save X_selected and selector to files
joblib.dump(X_selected, '../models/X_selected.pkl')

print("Feature engineering completed successfully.")
