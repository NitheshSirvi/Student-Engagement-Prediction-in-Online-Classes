import pandas as pd
import numpy as np
import os
from config.config import DATA_PATH

def generate_synthetic_data(num_records=2000):
    """Generates synthetic clickstream and interaction data."""
    print("Generating synthetic student data...")
    np.random.seed(42)
    
    data = {
        'time_on_platform_mins': np.random.normal(120, 45, num_records).clip(10, 300),
        'num_video_clicks': np.random.poisson(15, num_records),
        'quiz_score_avg': np.random.normal(75, 15, num_records).clip(0, 100),
        'forum_posts': np.random.poisson(2, num_records),
        'is_engaged': np.random.choice([0, 1], size=num_records, p=[0.3, 0.7])
    }
    
    df = pd.DataFrame(data)
    
    # Add some correlation: Higher engagement correlates with higher scores/time
    df.loc[df['is_engaged'] == 1, 'quiz_score_avg'] += 10
    df.loc[df['is_engaged'] == 1, 'time_on_platform_mins'] += 30
    df['quiz_score_avg'] = df['quiz_score_avg'].clip(0, 100)
    
    df.to_csv(DATA_PATH, index=False)
    print(f"Data saved to {DATA_PATH}")
    return df