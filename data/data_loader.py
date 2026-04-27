import pandas as pd
from config.config import DATA_PATH

def load_data():
    """Loads the dataset from the configured path."""
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print("Data file not found. Please run data generator first.")
        return None