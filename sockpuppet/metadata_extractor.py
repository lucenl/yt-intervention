"""
Extract metadata from YouTube videos for classification.
"""

import os
import json
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
import pandas as pd

class MetadataExtractor:
    """Extract metadata from YouTube videos using yt-dlp"""
    
    def __init__(self, output_dir, timeout=60, max_workers=10):
        """
        Initialize the extractor
        
        Args:
            output_dir: Directory to save metadata
            timeout: Maximum time (seconds) for video processing
            max_workers: Maximum concurrent workers
        """
        self.output_dir = output_dir
        self.metadata_dir = os.path.join(output_dir, "metadata")
        self.timeout = timeout
        self.max_workers = max_workers
        
        # Create directories
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
    
    def extract_metadata_batch(self, video_ids):
        """
        Extract metadata for multiple videos
        
        Args:
            video_ids: List of YouTube video IDs
            
        Returns:
            Path to the CSV file with metadata
        """
        self.logger.info(f"Extracting metadata for {len(video_ids)} videos")
        
        # Filter out already processed videos
        to_process = [vid for vid in video_ids if not os.path.exists(f'{self.metadata_dir}/{vid}.json')]
        to_process = list(set(to_process))  # Remove duplicates
        
        self.logger.info(f"{len(to_process)}/{len(video_ids)} videos need processing")
        
        if to_process:
            # Process videos
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self._process_video, vid) for vid in to_process]
                
                for future in tqdm(futures, desc="Processing videos"):
                    future.result()
                    
        # Extract and combine metadata
        self.logger.info("Creating combined metadata CSV")
        csv_path = self._create_metadata_csv(video_ids)
        
        return csv_path
    
    def _process_video(self, video_id):
        """Download metadata for single video"""
        output_path = f'{self.metadata_dir}/{video_id}.json'
        
        try:
            # Construct yt-dlp command to get metadata including transcript
            cmd = f'yt-dlp -J --write-auto-sub --sub-lang en --skip-download "https://youtube.com/watch?v={video_id}" > {output_path} 2>/dev/null'
            
            # Execute command with timeout
            result = subprocess.run(cmd, shell=True, timeout=self.timeout)
            
            # Check if file exists and has content
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                # Try to extract transcript separately if needed
                self._extract_transcript(video_id, output_path)
                return video_id
            else:
                self.logger.warning(f"Empty output for {video_id}")
                return None
                
        except subprocess.TimeoutExpired:
            self.logger.warning(f"Timeout for {video_id}")
            return None
        except Exception as e:
            self.logger.error(f"Error processing {video_id}: {e}")
            return None
    
    def _extract_transcript(self, video_id, json_path):
        """Extract transcript and add it to the metadata file"""
        try:
            # Read metadata file
            with open(json_path, 'r') as f:
                metadata = json.load(f)
                
            # Check if transcript already exists
            if 'transcript' in metadata:
                return
                
            # Try to extract transcript from subtitles
            transcript = ""
            
            # Process subtitles or automatic captions
            subtitles = metadata.get('subtitles', {})
            auto_captions = metadata.get('automatic_captions', {})
            
            # Try to get English subtitles first, then fall back to auto-captions
            sub_entries = subtitles.get('en', []) or auto_captions.get('en', [])
            
            for entry in sub_entries:
                if entry.get('ext') == 'vtt':
                    # Download the subtitle file
                    sub_url = entry.get('url')
                    if sub_url:
                        sub_path = f'{self.metadata_dir}/{video_id}.vtt'
                        cmd = f"curl -s '{sub_url}' > {sub_path}"
                        subprocess.run(cmd, shell=True, timeout=self.timeout)
                        
                        # Read and process subtitle file
                        if os.path.exists(sub_path):
                            with open(sub_path, 'r') as f:
                                lines = f.readlines()
                                
                            # Extract text lines (skip timestamps and position)
                            text_lines = []
                            for line in lines:
                                line = line.strip()
                                if line and not line.startswith('WEBVTT') and not line[0].isdigit() and not '-->' in line:
                                    text_lines.append(line)
                                    
                            transcript = ' '.join(text_lines)
                            
                            # Clean up subtitle file
                            os.remove(sub_path)
                            break
            
            # Add transcript to metadata and save
            metadata['transcript'] = transcript
            
            with open(json_path, 'w') as f:
                json.dump(metadata, f)
                
        except Exception as e:
            self.logger.error(f"Error extracting transcript for {video_id}: {e}")
    
    def _create_metadata_csv(self, video_ids):
        """Create CSV with metadata for specified video IDs"""
        records = []
        
        for video_id in video_ids:
            json_path = f'{self.metadata_dir}/{video_id}.json'
            
            if not os.path.exists(json_path):
                # Add minimal record if file doesn't exist
                records.append({
                    'links': f"https://youtube.com/watch?v={video_id}",
                    'video_id': video_id,
                    'channel': '',
                    'title': '',
                    'description': '',
                    'transcript': '',
                    'date': ''
                })
                continue
                
            try:
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
                    
                # Extract relevant fields
                record = {
                    'links': f"https://youtube.com/watch?v={video_id}",
                    'video_id': video_id,
                    'channel': metadata.get('channel', metadata.get('uploader', '')),
                    'title': metadata.get('title', ''),
                    'description': metadata.get('description', ''),
                    'transcript': metadata.get('transcript', ''),
                    'date': metadata.get('upload_date', '')
                }
                
                records.append(record)
                
            except Exception as e:
                self.logger.error(f"Error reading metadata for {video_id}: {e}")
                records.append({
                    'links': f"https://youtube.com/watch?v={video_id}",
                    'video_id': video_id,
                    'channel': '',
                    'title': '',
                    'description': '',
                    'transcript': '',
                    'date': ''
                })
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(records)
        csv_path = os.path.join(self.output_dir, "recommendations_metadata.csv")
        df.to_csv(csv_path, index=False)
        
        # Print summary
        self.logger.info(f"Created metadata CSV with {len(df)} records")
        
        return csv_path