import pandas as pd
import os

def clean_kaggle_dataset(input_csv_path, output_csv_path):
    print(f"Reading raw data from: {input_csv_path}...")
    
    try:
        # 1. Load the messy dataset
        df = pd.read_csv(input_csv_path)
        print(f"Original columns found: {list(df.columns)}")
        
        # ---------------------------------------------------------
        # 2. THE MAPPING DICTIONARY (You must edit this part!)
        # Change the text on the LEFT side of the colon to match the 
        # exact column names in your downloaded Kaggle file.
        # Keep the RIGHT side exactly as it is.
        # ---------------------------------------------------------
        column_mapping = {
            'ugly_time_column_name': 'time_on_platform_mins',
            'ugly_clicks_column_name': 'num_video_clicks',
            'ugly_score_column_name': 'quiz_score_avg',
            'ugly_forum_column_name': 'forum_posts'
        }
        
        # 3. Rename the columns
        df = df.rename(columns=column_mapping)
        
        # 4. Filter the dataset to ONLY keep the 4 columns our AI needs
        required_cols = ['time_on_platform_mins', 'num_video_clicks', 'quiz_score_avg', 'forum_posts']
        
        # Check if the renaming worked before filtering
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"\n❌ ERROR: Could not find these required columns after renaming: {missing_cols}")
            print("Did you update the 'column_mapping' dictionary correctly?")
            return
            
        final_df = df[required_cols].copy()
        
        # 5. Clean missing values (Fill empty cells with the column average)
        final_df = final_df.fillna(final_df.mean())
        
        # 6. Save the clean file
        final_df.to_csv(output_csv_path, index=False)
        print(f"\n✅ Success! Clean data saved to: {output_csv_path}")
        print(f"Ready to upload to the dashboard. Final columns: {list(final_df.columns)}")
        
    except FileNotFoundError:
        print(f"❌ ERROR: Could not find the file at {input_csv_path}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

# --- Execution ---
if __name__ == "__main__":
    # Point this to wherever you saved your downloaded Kaggle file
    RAW_FILE = "data/my_messy_kaggle_file.csv"  
    
    # This is the new, clean file it will generate
    CLEAN_FILE = "data/ready_for_dashboard.csv" 
    
    clean_kaggle_dataset(RAW_FILE, CLEAN_FILE)