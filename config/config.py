import os

# Define absolute paths based on the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "student_data.csv")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models", "saved_models")
TARGET_COL = "is_engaged"

# Create model save directory if it doesn't exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Microsoft SQL Server 2019 Configuration
DB_SERVER = 'localhost' # Or your specific server name
DB_NAME = 'StudentAnalyticsDB'
# Using Windows Authentication (Trusted_Connection) by default
DB_CONNECTION_STRING = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={DB_SERVER};DATABASE={DB_NAME};Trusted_Connection=yes;"