import pyodbc
from datetime import datetime
from config.config import DB_CONNECTION_STRING

def log_prediction_to_sql(time_spent, clicks, quiz_score, forum_posts, prediction, probability):
    """Logs the live prediction data into Microsoft SQL Server 2019."""
    try:
        # Establish connection
        conn = pyodbc.connect(DB_CONNECTION_STRING)
        cursor = conn.cursor()
        
        # SQL Insert Query
        insert_query = """
        INSERT INTO EngagementLogs 
        (Timestamp, TimeOnPlatform, VideoClicks, QuizScore, ForumPosts, PredictionResult, ConfidenceScore)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        # Execute Query
        timestamp = datetime.now()
        cursor.execute(insert_query, 
                       (timestamp, time_spent, clicks, quiz_score, forum_posts, int(prediction), float(probability)))
        
        # Commit and close
        conn.commit()
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"Database Error: {e}")
        # If the database isn't set up yet on your machine, it will just fail silently 
        # so the app doesn't crash during your presentation.
        return False