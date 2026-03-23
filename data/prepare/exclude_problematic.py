import json
import pandas as pd
import ast

def extract_non_problematic_videos(input_file, output_file):
    """
    Extract all non-problematic videos from the JSON file and save to a CSV.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path to the output CSV file
    """
    try:
        # Load the JSON data
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract all non-problematic videos
        non_problematic_videos = []
        
        for user in data['users']:
            for video in user['videos']:
                if not video['is_problematic']:
                    non_problematic_videos.append(video['videoId'])
        
        # Remove duplicates
        unique_videos = list(set(non_problematic_videos))
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame({'videoId': unique_videos})
        df.to_csv(output_file, index=False)
        
        print(f"Extracted {len(unique_videos)} unique non-problematic videos")
        print(f"Saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")

# Alternative version for CSV input
def extract_non_problematic_from_csv(input_file, output_file):
    """
    Extract all non-problematic videos from the CSV file and save to a CSV.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to the output CSV file
    """
    try:
        # Read CSV with pandas
        df = pd.read_csv(input_file, dtype={'visaId': str, 'watch_history': str})
        
        non_problematic_videos = []
        
        for _, row in df.iterrows():
            watch_history = eval(row['watch_history'], {'Timestamp': pd.Timestamp})
            
            for entry in watch_history:
                if not entry["is_problematic"]:
                    non_problematic_videos.append(entry["videoId"])
        
        # Remove duplicates
        unique_videos = list(set(non_problematic_videos))
        
        # Create DataFrame and save to CSV
        result_df = pd.DataFrame({'videoId': unique_videos})
        result_df.to_csv(output_file, index=False)
        
        print(f"Extracted {len(unique_videos)} unique non-problematic videos")
        print(f"Saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Choose which function to use based on your input format
    # If your input is JSON:
    extract_non_problematic_videos("user_watch_histories_with_problematic.json", "non_problematic_videos.csv")
    
    # If your input is CSV:
    # extract_non_problematic_from_csv("user_watch_histories_with_problematic.csv", "non_problematic_videos.csv")