import json
import csv
from datetime import datetime
import pandas as pd
import ast

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle pandas Timestamp objects"""
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super().default(obj)

def convert_timestamp(timestamp_str):
    """Convert timestamp string to ISO format string."""
    try:
        clean_ts = timestamp_str.replace("Timestamp('", "").replace("')", "")
        dt = datetime.strptime(clean_ts, "%Y-%m-%d %H:%M:%S.%f")
        return dt.isoformat()
    except Exception as e:
        return timestamp_str

def process_video_data(visa_id, watch_history_str):
    """
    Process the video watch history for a specific visa ID.
    
    Args:
        visa_id (str): The visa ID
        watch_history_str (str): String representation of the watch history list
        
    Returns:
        dict: Processed data in JSON format
    """
    # Convert string to Python object
    watch_history = eval(watch_history_str, {'Timestamp': pd.Timestamp})
    
    # Group videos by videoId
    videos_by_id = {}
    for entry in watch_history:
        video_id = entry["videoId"]
        if video_id not in videos_by_id:
            videos_by_id[video_id] = {
                "videoId": video_id,
                "date": entry["date"],
                "is_problematic": entry["is_problematic"],
                "view_count": 1,  # Initialize view count
                "metadata": entry["metadata"],
            }
        else:
            videos_by_id[video_id]["view_count"] += 1  # Increment view count
    
    # Create final output structure
    output = {
        "visa_id": visa_id,
        "videos": list(videos_by_id.values()),
        "total_unique_videos": len(videos_by_id),
        "total_views": len(watch_history)
    }
    
    return output

def convert_csv_to_json(input_file, output_file):
    """Convert CSV file with visaId and watch_history to JSON format."""
    # Read CSV with pandas
    df = pd.read_csv(input_file, dtype={'visaId': str, 'watch_history': str})
    
    all_users_data = []
    for _, row in df.iterrows():
        processed_data = process_video_data(row['visaId'], row['watch_history'])
        all_users_data.append(processed_data)
    
    # Save to JSON file using custom encoder
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "users": all_users_data,
            "total_users": len(all_users_data),
        }, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
    
    print(f"Data has been processed and saved to {output_file}")

# Example usage:
if __name__ == "__main__":
    convert_csv_to_json("user_watch_histories_with_problematic.csv", "user_watch_histories_with_problematic.json")