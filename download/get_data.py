import argparse
import os
import json
import csv
import pandas as pd
import concurrent.futures
from ytdriver.Video import Video, VideoUnavailableException

def make_url(videoId):
    return 'https://youtube.com/watch?v=' + str(videoId)

def save_metadata_to_json(metadata, output_dir, filename):
    """Save video metadata to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    # Convert metadata object to dictionary (excluding the full video_json)
    metadata_dict = {key: value for key, value in metadata.__dict__.items() 
                     if key != 'video_json'}
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
    
    return output_path

def process_video(video_id, puppet_id, harmful_percentage, output_dir):
    """Worker function that fetches metadata for a single video,
       saves its JSON file, and returns a dictionary with the row data.
    """
    try:
        video = Video(None, make_url(video_id))
        metadata = video.get_metadata()
        filename = f"{metadata.id}.json"
        output_path = save_metadata_to_json(metadata, output_dir, filename)
        print(f"✓ Saved metadata for '{metadata.title}' to {output_path}")
        row = {
            'link': getattr(metadata, 'webpage_url', ''),
            'video_id': getattr(metadata, 'id', ''),
            'channel_name': getattr(metadata, 'uploader', ''),
            'title': getattr(metadata, 'title', ''),
            'description': getattr(metadata, 'description', ''),
            'date': getattr(metadata, 'upload_date', ''),
            'duration': getattr(metadata, 'duration', ''),
            'views': getattr(metadata, 'view_count', ''),
            'puppet_id': puppet_id,
            'harmful_percentage': harmful_percentage
        }
        return {"row": row, "error": None}
    except VideoUnavailableException:
        return {"row": None, "error": {'url': make_url(video_id), 'reason': 'Video unavailable'}}
    except Exception as e:
        return {"row": None, "error": {'url': make_url(video_id), 'reason': str(e)}}

def process_json(input_file, output_dir, workers):
    output_file = 'metadata.csv'
    os.makedirs(output_dir, exist_ok=True)
    
    # Read existing CSV (if any) to skip already processed videos.
    processed_ids = set()
    if os.path.exists(output_file):
        try:
            df_existing = pd.read_csv(output_file, encoding='utf-8')
            processed_ids = set(df_existing['video_id'].astype(str))
            print(f"Found {len(processed_ids)} already processed videos in {output_file}")
        except Exception as e:
            print(f"Could not read existing CSV: {e}")

    # Open CSV file in append mode.
    file_exists = os.path.exists(output_file)
    # Write header if CSV does not exist.
    header = ['link', 'video_id', 'channel_name', 'title', 'description',
                  'date', 'duration', 'views', 'puppet_id', 'harmful_percentage']
    
    errors = []
    total_videos = 0
    successful = 0
    failed = 0

    tasks = []
    
    rows = []
    # Build the list of tasks from the JSON file.
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for record in data:
        puppet_id = record.get('puppet_id', '')
        harmful_percentage = record.get('harmful_percentage', '')
        for key in ['homepage_recs', 'upnext_recs']:
            video_ids = record.get(key, [])
            for video_id in video_ids:
                metadata_filename = f"{video_id}.json"
                metadata_path = os.path.join(output_dir, metadata_filename)
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                    except Exception as e:
                        print(f"Error reading metadata file: {e}")
                        continue
                    # Build a row from metadata combined with input record information.
                    row = [
                        metadata.get('webpage_url', ''),
                        metadata.get('id', ''),
                        metadata.get('channel_id', ''),
                        metadata.get('title', ''),
                        metadata.get('description', ''),
                        metadata.get('upload_date', ''),
                        metadata.get('duration', ''),
                        metadata.get('view_count', ''),
                        puppet_id,
                        harmful_percentage
                    ]
                    rows.append(row)
                else:
                    print(f"Metadata file not found: {metadata_path}")
                    
    print(f"Total videos to process: {len(rows)}")
    with open(output_file, 'a', encoding='utf-8', newline='') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(header)
        writer.writerows(rows)
    
    print(f"CSV file generated at: {output_file}")

    if errors:
        with open("errors.json", 'w', encoding='utf-8') as f:
            json.dump(errors, f, indent=2)
        print("\nError details saved to errors.json")
    
    print(f"\nMetadata saved to {output_file}")
    print(f"Total videos attempted: {total_videos}")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")

def main():
    parser = argparse.ArgumentParser(
        description='Process YouTube video metadata from JSON records using multiple threads with line-by-line CSV writing.'
    )
    parser.add_argument('input_file', help='Input JSON file with video records')
    parser.add_argument('-o', '--output-dir', default='video_metadata',
                        help='Directory to save metadata files (default: video_metadata)')
    parser.add_argument('--workers', type=int, default=2,
                        help='Number of worker threads to use (default: 4)')
    args = parser.parse_args()
    
    process_json(args.input_file, args.output_dir, args.workers)

if __name__ == "__main__":
    main()
