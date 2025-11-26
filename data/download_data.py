import pandas as pd
from sklearn.datasets import fetch_california_housing
import warnings
warnings.filterwarnings('ignore')

# Load California Housing dataset
print("Loading California Housing dataset...")
housing = fetch_california_housing()

# Create DataFrame
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['PRICE'] = housing.target

# Save to CSV
df.to_csv('data/raw_data.csv', index=False)
print(f"Dataset saved successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nDataset Info:")
print(f"- Total samples: {df.shape[0]}")
print(f"- Total features: {df.shape[1] - 1}")
print(f"- Target variable: House Price (in $100,000s)")