import os
import json
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
import requests
import time
from typing import List, Dict, Optional
import re

API_KEY = 'AIzaSyBGdcHykAwbhEND9MtC-3ZAYhiXXyfV1As'

class MetadataExtractor:
    """
    Extract metadata from YouTube videos using YouTube Data API v3 and transcripts using yt-dlp.
    Uses videos.list (contentDetails) and captions.list to pre-check subtitle availability.
    Stores results in JSON format for integration with experiment logging.
    """

    def __init__(self, output_dir: str, api_key: Optional[str] = None, timeout: int = 60, max_workers: int = 4, api_key=API_KEY):
        """
        Initialize the extractor.

        Args:
            output_dir: Directory to save metadata and transcripts
            api_key: YouTube Data API v3 key (defaults to environment variable YOUTUBE_API_KEY)
            timeout: Maximum time (seconds) for transcript processing
            max_workers: Maximum concurrent workers for transcript extraction
        """
        self.output_dir = output_dir
        self.api_key = api_key
        self.timeout = timeout
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
        self.videos_url = "https://www.googleapis.com/youtube/v3/videos"

    def extract_metadata_batch(self, video_ids: List[str], puppet_id: str) -> List[Dict]:
        """
        Extract metadata and transcripts for multiple videos and return as a list of dictionaries.

        Args:
            video_ids: List of YouTube video IDs
            puppet_id: Identifier for organizing metadata

        Returns:
            List of dictionaries containing metadata for each video
        """
        metadata_dir = os.path.join(self.output_dir, "metadata", puppet_id)
        os.makedirs(metadata_dir, exist_ok=True)
        self.logger.info(f"Extracting metadata and transcripts for {len(video_ids)} videos")

        # Remove duplicates and validate video IDs
        video_ids = list(set([vid for vid in video_ids]))
        if not video_ids:
            self.logger.warning("No valid video IDs provided")
            return []

        # Fetch metadata and subtitle availability using YouTube Data API
        api_metadata = self._fetch_api_metadata_and_subtitles(video_ids)

        # Filter videos that need transcript processing
        to_process = [vid for vid in video_ids]
        self.logger.info(f"{len(to_process)}/{len(video_ids)} videos need transcript processing")

        # Process transcripts concurrently using yt-dlp
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._process_transcript, vid, metadata_dir)
                for vid in to_process
            ]
            for future in tqdm(futures, desc="Processing transcripts"):
                future.result()

        # Collect and merge metadata
        self.logger.info("Collecting metadata into list")
        metadata_list = self._collect_metadata(video_ids, metadata_dir, api_metadata)

        return metadata_list

    def _fetch_api_metadata_and_subtitles(self, video_ids: List[str]) -> tuple[Dict[str, Dict], Dict[str, bool]]:
        """
        Fetch metadata and subtitle availability using YouTube Data API v3 (videos.list).

        Args:
            video_ids: List of YouTube video IDs

        Returns:
            Tuple of (metadata dictionary, subtitle availability dictionary)
        """
        metadata_dict = {}
        subtitle_availability = {vid: False for vid in video_ids}
        batch_size = 4  # API limit
        for i in range(0, len(video_ids), batch_size):
            batch_ids = video_ids[i:i + batch_size]
            video_ids_csv = ','.join(batch_ids)
            url = f"{self.videos_url}?part=snippet,contentDetails&id={video_ids_csv}&key={self.api_key}"

            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()

                for item in data.get('items', []):
                    vid = item['id']
                    metadata_dict[vid] = {
                        'video_id': vid,
                        'channel': item['snippet'].get('channelTitle', ''),
                        'title': item['snippet'].get('title', ''),
                        'description': item['snippet'].get('description', ''),
                        'date': item['snippet'].get('publishedAt', '')  # YYYY-MM-DD
                    }

            except requests.exceptions.HTTPError as e:
                self.logger.error(f"Failed to fetch metadata for batch: {batch_ids}")

            except requests.exceptions.RequestException as e:
                self.logger.warning(f"API request failed: {e}")

        return metadata_dict

    def _process_transcript(self, video_id: str, metadata_dir: str) -> Optional[str]:
        """
        Extract transcript for a single video using yt-dlp.

        Args:
            video_id: YouTube video ID
            metadata_dir: Directory to save metadata files

        Returns:
            Video ID if successful, None otherwise
        """
        transcript_dir = os.path.join(metadata_dir, "transcripts")
        os.makedirs(transcript_dir, exist_ok=True)
        output_path = os.path.join(transcript_dir, f"{video_id}.txt")

        try:
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return video_id  # Transcript already exists

            # Streamlined command to get transcript
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

            # Check if transcript file was created
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return video_id
            else:
                self.logger.info(f"No transcript found for {video_id}")
                return None

        except subprocess.TimeoutExpired:
            self.logger.warning(f"Timeout for transcript of {video_id}")
            return None
        except Exception as e:
            self.logger.error(f"Error processing transcript for {video_id}: {e}")
            return None

    def _collect_metadata(self, video_ids: List[str], metadata_dir: str, api_metadata: Dict[str, Dict]) -> List[Dict]:
        """
        Collect and merge metadata from API and transcript files into a list of dictionaries.

        Args:
            video_ids: List of YouTube video IDs
            metadata_dir: Directory where metadata is stored
            api_metadata: Dictionary of API-fetched metadata

        Returns:
            List of metadata dictionaries
        """
        metadata_list = []

        for video_id in video_ids:
            json_path = os.path.join(metadata_dir, f"{video_id}.json")
            transcript_path = os.path.join(metadata_dir, "transcripts", f"{video_id}.txt")

            # Start with API metadata or default
            record = api_metadata.get(video_id, {
                'video_id': video_id,
                'channel': '',
                'title': '',
                'description': '',
                'date': ''
            })

            # Try to read existing transcript from file
            try:
                if os.path.exists(transcript_path) and os.path.getsize(transcript_path) > 0:
                    with open(transcript_path, 'r', encoding='utf-8') as f:
                        transcript = f.read().strip()
                        transcript = re.sub(r'\s+', ' ', transcript).strip()
                        record['transcript'] = transcript
                else:
                    record['transcript'] = ""
            except Exception as e:
                self.logger.error(f"Error reading transcript for {video_id}: {e}")

            # Save or update JSON file
            try:
                with open(json_path, 'w') as f:
                    json.dump(record, f, indent=4)
                metadata_list.append(record)
            except Exception as e:
                self.logger.error(f"Error saving metadata for {video_id}: {e}")
                metadata_list.append(record)

        self.logger.info(f"Collected metadata for {len(metadata_list)} videos")
        return metadata_list