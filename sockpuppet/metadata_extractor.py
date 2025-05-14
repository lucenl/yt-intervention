"""
YouTube Video Metadata Extraction Module

Extract metadata and transcripts from YouTube videos for classification,
storing results in JSON format for integration with experiment logging.
"""

import os
import json
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
import pandas as pd
import re

class MetadataExtractor:
    """
    Extract metadata and transcripts from YouTube videos using yt-dlp.
    """

    def __init__(self, output_dir, timeout=60, max_workers=4):
        """
        Initialize the extractor.

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
        Extract metadata and transcripts for multiple videos and return as a list of dictionaries.

        Args:
            video_ids: List of YouTube video IDs

        Returns:
            List of dictionaries containing metadata for each video
        """
        self.logger.info(f"Extracting metadata and transcripts for {len(video_ids)} videos")
        
        # Filter out already processed videos
        to_process = [vid for vid in video_ids if not os.path.exists(f'{self.metadata_dir}/{vid}.json')]
        to_process = list(set(to_process))  # Remove duplicates
        
        self.logger.info(f"{len(to_process)}/{len(video_ids)} videos need processing")
        
        if to_process:
            # Process videos concurrently
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self._process_video, vid) for vid in to_process]
                
                for future in tqdm(futures, desc="Processing videos"):
                    future.result()
                    
        # Collect metadata into a list
        self.logger.info("Collecting metadata into list")
        metadata_list = self._collect_metadata(video_ids)
        
        return metadata_list

    def _process_video(self, video_id):
        """
        Download metadata for a single video.

        Args:
            video_id: YouTube video ID

        Returns:
            Video ID if successful, None otherwise
        """
        output_path = f'{self.metadata_dir}/{video_id}.json'
        
        try:
            # Construct yt-dlp command to get metadata
            cmd = f'yt-dlp -J --write-auto-sub --sub-lang en --skip-download "https://youtube.com/watch?v={video_id}" > {output_path} 2>/dev/null'
            
            # Execute command with timeout
            result = subprocess.run(cmd, shell=True, timeout=self.timeout)
            
            # Check if file exists and has content
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                # Extract transcript using the new method
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
        """
        Extract transcript using optimized yt-dlp command and add it to the metadata file.

        Args:
            video_id: YouTube video ID
            json_path: Path to the JSON metadata file
        """
        try:
            # Read metadata file
            with open(json_path, 'r') as f:
                metadata = json.load(f)
                
            # Check if transcript already exists
            if 'transcript' in metadata and metadata['transcript']:
                return
                
            # Extract transcript using the optimized method
            transcript_dir = os.path.join(self.metadata_dir, "transcripts")
            os.makedirs(transcript_dir, exist_ok=True)
            
            output_path = os.path.join(transcript_dir, f"{video_id}.txt")
            
            # Check if transcript file already exists
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                with open(output_path, 'r', encoding='utf-8') as f:
                    transcript = f.read().strip()
                    transcript = re.sub(r'\s+', ' ', transcript)
            else:
                # Streamlined command to get transcript and create a single continuous line
                temp_srt = os.path.join(transcript_dir, f"temp_{video_id}.en.srt")
                cmd = (
                    f"yt-dlp --skip-download --write-subs --write-auto-subs --sub-lang en "
                    f"--sub-format ttml --convert-subs srt --output '{transcript_dir}/temp_{video_id}.%(ext)s' "
                    f"https://youtube.com/watch?v={video_id} > /dev/null 2>&1 && "
                    f"cat '{temp_srt}' 2>/dev/null | "
                    f"grep -v '^[0-9]\\+$' | grep -v '^[0-9][0-9]:[0-9][0-9]:[0-9][0-9]' | "
                    f"sed 's/<[^>]*>//g' | grep -v '^$' | tr '\\n' ' ' | sed 's/\\s\\+/ /g' > '{output_path}' && "
                    f"rm -f '{temp_srt}'"
                )
                
                # Execute command with timeout
                subprocess.run(cmd, shell=True, timeout=self.timeout)
                
                # Check if transcript file was created and has content
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    with open(output_path, 'r', encoding='utf-8') as f:
                        transcript = f.read().strip()
                        transcript = re.sub(r'\s+', ' ', transcript)
                else:
                    transcript = ""
                    # self.logger.warning(f"No transcript found for {video_id}")
            
            # Add transcript to metadata and save
            metadata['transcript'] = transcript
            
            with open(json_path, 'w') as f:
                json.dump(metadata, f)
                
        except Exception as e:
            self.logger.error(f"Error extracting transcript for {video_id}: {e}")
            # Ensure metadata file isn't corrupted
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            metadata['transcript'] = ""
            with open(json_path, 'w') as f:
                json.dump(metadata, f)

    def _collect_metadata(self, video_ids):
        """
        Collect metadata for specified video IDs into a list of dictionaries.

        Args:
            video_ids: List of YouTube video IDs

        Returns:
            List of metadata dictionaries
        """
        metadata_list = []
        
        for video_id in video_ids:
            json_path = f'{self.metadata_dir}/{video_id}.json'
            
            if not os.path.exists(json_path):
                # Add minimal record if file doesn't exist
                metadata_list.append({
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
                    
                record = {
                    'links': f"https://youtube.com/watch?v={video_id}",
                    'video_id': video_id,
                    'channel': metadata.get('uploader', ''),
                    'title': metadata.get('title', ''),
                    'description': metadata.get('description', ''),
                    'transcript': metadata.get('transcript', ''),
                    'date': metadata.get('upload_date', '')
                }
                
                metadata_list.append(record)
                
            except Exception as e:
                self.logger.error(f"Error reading metadata for {video_id}: {e}")
                metadata_list.append({
                    'links': f"https://youtube.com/watch?v={video_id}",
                    'video_id': video_id,
                    'channel': '',
                    'title': '',
                    'description': '',
                    'transcript': '',
                    'date': ''
                })
        
        self.logger.info(f"Collected metadata for {len(metadata_list)} videos")
        return metadata_list