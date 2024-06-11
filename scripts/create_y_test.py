import pandas as pd
from sklearn.model_selection import train_test_split

# Load the original dataset
data = pd.read_csv('../data/heart.csv')

# Extract the target variable
y = data['output']

# Perform any necessary preprocessing on y (e.g., encoding categorical variables)

# Split the data into features (X) and target variable (y)
X = data.drop('output', axis=1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save y_test to a pickle file
pd.to_pickle(y_test, 'data/y_test.pkl')
