import pandas as pd
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to the CSV file
csv_file = os.path.join(current_dir, '../data/heart.csv')

# Load data from CSV file
data = pd.read_csv(csv_file)

# Display the first few rows of the dataframe
print(data.head())
