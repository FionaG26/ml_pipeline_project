from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Load data from CSV file
data = pd.read_csv('../data/heart.csv')

# Split data into features and target
X = data.drop('output', axis=1)
y = data['output']

# Ensure X and y have the same number of samples
X = X.iloc[:len(y)]

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

# Split data into training and test sets after preprocessing
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Ensure the 'models' directory exists
os.makedirs('../models', exist_ok=True)

# Save preprocessed data and split datasets to files
pd.to_pickle(X_preprocessed, '../models/X_preprocessed.pkl')
pd.to_pickle(y_test, '../models/y_test.pkl')

print("Data preprocessing completed.")
