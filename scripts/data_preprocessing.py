from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
import os

# Load data from CSV file
data = pd.read_csv('../data/heart.csv')

# Split data into features and target
X = data.drop('output', axis=1)
y = data['output']

# Identify numerical and categorical columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Define transformations for numerical and categorical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformations into a preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

# Preprocess the features
X_preprocessed = preprocessor.fit_transform(X)

# Ensure the 'data' directory exists
os.makedirs('data', exist_ok=True)

# Save preprocessed data to file
pd.to_pickle(X_preprocessed, 'data/X_preprocessed.pkl')

print("Data preprocessing completed.")
