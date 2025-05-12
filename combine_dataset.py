import pandas as pd

# Load your existing recipes.csv into a pandas DataFrame
existing_data = pd.read_csv('recipes.csv')
# Load the new dataset from Kaggle
new_data = pd.read_csv('path_to_new_kaggle_dataset.csv')

# Print the first few rows of the new data to verify its structure
print(new_data.head())

# Print the first few rows to verify the data
print(existing_data.head())
