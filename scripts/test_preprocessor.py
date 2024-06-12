import joblib
import pandas as pd

# Load the preprocessor
preprocessor = joblib.load('models/preprocessor.pkl')

# Define some sample data for testing
sample_data = [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]

# Convert the sample data to a DataFrame
sample_df = pd.DataFrame(sample_data)

# Preprocess the sample data using the loaded preprocessor
try:
    preprocessed_data = preprocessor.transform(sample_df)
    print("Preprocessed data:")
    print(preprocessed_data)
except Exception as e:
    print("Error occurred during preprocessing:", e)
