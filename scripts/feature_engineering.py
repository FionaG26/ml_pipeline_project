import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif


# Load preprocessed data from file
X_preprocessed = pd.read_pickle('../models/X_preprocessed.pkl')

# Load the original data to extract the target variable
data = pd.read_csv('../data/heart.csv')

# Define the target variable
y = data['output']

# Feature selection
selector = SelectKBest(score_func=f_classif, k='all')  # Adjust 'k' as needed
X_selected = selector.fit_transform(X_preprocessed, y)

# Save X_selected to a file
pd.to_pickle(X_selected, '../models/X_selected.pkl')

# Print a message indicating successful execution
print("Feature engineering completed successfully.")
