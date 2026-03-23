from ytdriver import YTDriver, Video, VideoUnavailableException
from selenium.common.exceptions import WebDriverException
import argparse
import pandas as pd
import os
import json
import time


def parse_args():
    """Parse command line arguments for video verification"""
    parser = argparse.ArgumentParser(description='Verify YouTube video availability')
    parser.add_argument('--videos', required=True, help='Path to CSV file with harmful videos')
    parser.add_argument('--output', default='./verified/videos.csv', help='Directory to save verified video lists')
    parser.add_argument('--batch-size', type=int, default=1000, help='Number of videos to verify in each batch')
    parser.add_argument('--start_idx', type=int, default=0, help='Starting index in the video list')
    parser.add_argument('--end_idx', type=int, default=None, help='Ending index (exclusive) in the video list')
    parser.add_argument('--worker', type=int, help='Worker for parallel processing')
    return parser.parse_args()


def make_url(videoId):
    return 'https://youtube.com/watch?v=' + str(videoId)

def verify_video_availability(videos, batch_size, start_idx, end_idx):
    """
    Verify availability of videos in harmful and harmless pools
    
    Args:
        harmful_pool: List of harmful video IDs
        harmless_pool: List of harmless video IDs
        batch_size: Number of videos to check at once before restarting driver
        
    Returns:
        verified_harmful: List of available harmful videos
        verified_harmless: List of available harmless videos
    """
    if end_idx is None:
        end_idx = len(videos)
    
    # Ensure indices are within range
    start_idx = max(0, min(start_idx, len(videos)))
    end_idx = max(start_idx, min(end_idx, len(videos)))
    
    videos = videos[start_idx:end_idx]
    print(f"Verifying {len(videos)} videos from index {start_idx} to {end_idx}")
    verified_videos = []
    unavailable_videos = []
    
    # Process in batches to avoid browser crashes
    def check_batch(videos_batch):
        driver = YTDriver(headless=True, verbose=True, use_virtual_display=False)
        batch_verified = []
        batch_unavailable = []
        
        try:
            for videoId in videos_batch:
                try:
                    # Create a Video object
                    video = Video(None, make_url(videoId))
                    driver._YTDriver__click_video(video)
                    driver._YTDriver__check_video_availability()
                    # If we get here, video is available
                    batch_verified.append(videoId)
                    print(f"Verified video: {videoId}")
                    
                except (VideoUnavailableException, WebDriverException) as e:
                    batch_unavailable.append(videoId)
                    print(f"Skipping unavailable video {videoId}: {str(e)}")
                    continue
                time.sleep(0.5)  # Small delay to avoid overwhelming the server
        finally:
            # Always close the driver
            driver.close()
            
        return batch_verified, batch_unavailable
    
    # Process videos in batches
    total_batches = (len(videos) + batch_size - 1) // batch_size
    
    for i in range(0, len(videos), batch_size):
        batch_num = i // batch_size + 1
        batch = videos[i:i + batch_size]
        
        print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch)} videos)")
        
        batch_verified, batch_unavailable = check_batch(batch)
        verified_videos.extend(batch_verified)
        unavailable_videos.extend(batch_unavailable)
        
        # Basic progress report
        print(f"Batch {batch_num} complete. Running totals: {len(verified_videos)} verified, {len(unavailable_videos)} unavailable")
    
    print(f"\nVerification complete for indices {start_idx}-{end_idx-1}")
    print(f"Total videos processed: {len(verified_videos) + len(unavailable_videos)}")
    print(f"Verified videos: {len(verified_videos)}")
    print(f"Unavailable videos: {len(unavailable_videos)}")
    
    return verified_videos, unavailable_videos

   
if __name__ == "__main__":
    args = parse_args()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(args.output)
        
    # Load harmful and harmless video IDs from CSV
    df = pd.read_csv(args.videos)
    videos = df['videoId'].tolist()
    
    # Verify video availability
    verified_videos, unavailable_videos = verify_video_availability(videos, args.batch_size, start_idx=args.start_idx, end_idx=args.end_idx)
    
    # Save to separate CSV files for clarity
    output_base = os.path.splitext(args.output)[0]
    pd.DataFrame({'videoId': verified_videos}).to_csv(f"{output_base}_harm_verified_{args.worker}.csv", index=False)
    pd.DataFrame({'videoId': unavailable_videos}).to_csv(f"{output_base}_harm_unavailable_{args.worker}.csv", index=False)
    
    print(f"Saved {len(verified_videos)} verified videos to {output_base}_verified_{args.worker}.csv")
    print(f"Saved {len(unavailable_videos)} unavailable videos to {output_base}_unavailable_{args.worker}.csv")