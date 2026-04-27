import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config.config import TARGET_COL

def prepare_data(df):
    """Cleans data, handles missing values, and applies standard scaling."""
    
    # --- NEW: Handling Missing Values (Slide 5 Requirement) ---
    print("Cleaning dataset and handling missing values...")
    # Fill missing numerical values with the median of that column
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Drop any remaining rows with critical missing data
    df = df.dropna()
    # ----------------------------------------------------------

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns