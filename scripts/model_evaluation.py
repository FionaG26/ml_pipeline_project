from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os

# Load data from CSV file
data = pd.read_csv('../data/heart.csv')

# Split data into features and target
X = data.drop('output', axis=1)
y = data['output']

# Ensure X and y have the same number of samples
X = X.iloc[:len(y)]

# Define preprocessing steps
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, X.select_dtypes(include=['int64', 'float64']).columns),
    ('cat', categorical_transformer, X.select_dtypes(include=['object']).columns)
])

# Preprocess the features
X_preprocessed = preprocessor.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Load the trained model from the saved file
best_model = joblib.load('../models/best_model.pkl')

# Predictions
y_pred = best_model.predict(X_test)

# Evaluation
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))
