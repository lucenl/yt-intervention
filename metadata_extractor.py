# metadata_service.py
import os
import json
import logging
import requests
import redis
from flask import Flask, request, jsonify

# Logging setup
LOCAL_LOG_DIR = "./local_logs"
os.makedirs(LOCAL_LOG_DIR, exist_ok=True)
log_file = os.path.join(LOCAL_LOG_DIR, "metadata_service.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a',
    force=True
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)


KEYS_ENV = 'AIzaSyBGdcHykAwbhEND9MtC-3ZAYhiXXyfV1As, AIzaSyCBkf4tEdjaUzczm6cWcolDoZhzD6IQXQg, AIzaSyAW88VfM3okyJyv3y5AqsFAWWaG7VoU7RA'


class MetadataExtractor:
    """
    Redis-backed YouTube metadata cache exposing Flask routes directly on the app.
    Supports rotating through a pool of YouTube API keys.
    """

    def __init__(self, redis_url='redis://localhost:6379/0', api_keys=None):
        # Read API keys from env or parameter (comma-separated)
        keys_env = KEYS_ENV
        self.api_keys = [k.strip() for k in keys_env.split(',') if k.strip()]
        if not self.api_keys:
            raise ValueError('Provide at least one YouTube API key via YOUTUBE_API_KEYS')
        self._key_index = 0

        # Redis connection
        self.REDIS_URL = redis_url
        self.cache = redis.Redis.from_url(self.REDIS_URL, decode_responses=True)

    def get_redis_client(self):
        """Return the Redis client for external use."""
        return self.cache
    
    def _get_next_key(self):
        key = self.api_keys[self._key_index]
        self._key_index = (self._key_index + 1) % len(self.api_keys)
        return key

    def _fetch_metadata(self, video_ids):
        """Fetch metadata and transcripts from YouTube Data API in bulk."""
        key = self._get_next_key()
        params = {
            'part': 'snippet',
            'id': ','.join(video_ids),
            'key': key
        }
        resp = requests.get('https://www.googleapis.com/youtube/v3/videos', params=params)
        resp.raise_for_status()
        items = resp.json().get('items', [])

        results = []
        for item in items:
            vid = item.get('id')
            logging.info("vid: ", vid)
            snippet = item.get('snippet', {})
            data = {
                'video_id': vid,
                'title': snippet.get('title', ''),
                'description': snippet.get('description', '')
            }
            results.append(data)
        logging.info(f"Fetched metadata for {len(results)} videos")
        return results

    def _save_metadata_to_file(self, items):
        all_metadata = {}
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                try:
                    all_metadata = json.load(f)
                except json.JSONDecodeError:
                    logging.warning(f"Corrupted {self.metadata_file}, starting fresh")
        
        for item in items:
            all_metadata[item['video_id']] = item
        
        with open(self.metadata_file, 'w') as f:
            json.dump(all_metadata, f, indent=4)
        logging.info(f"Saved {len(items)} new metadata entries to {self.metadata_file}")
        
    def register(self, app: Flask):
        """Attach /metadata endpoints directly to the given Flask app."""

        @app.route('/metadata', methods=['GET'])
        def get_metadata():
            ids = request.args.get('ids', '')
            video_ids = [v for v in ids.split(',') if v]
            if not video_ids:
                return jsonify({'error': 'No video IDs provided'}), 400

            # Redis hash 'metadata' stores video_id->JSON
            cached = self.cache.hgetall('metadata')
            logging.info(f"Cached metadata with length: {len(cached)}")
            missing = [vid for vid in video_ids if vid not in cached]
            logging.info(f"Missing: {missing}")

            if missing:
                logging.info(f"Fetching metadata for missing vidoes: {len(missing)}")
                try:
                    new_items = self._fetch_metadata(missing)
                    for item in new_items:
                        self.cache.hset('metadata', item['video_id'], json.dumps(item))
                    # Update local copy
                    for item in new_items:
                        cached[item['video_id']] = json.dumps(item)
                    logging.info(f"Updated metadata cache with {len(new_items)} new items")
                except Exception as e:
                    logging.error(f'Error fetching metadata for {missing}: {e}')

            # Return results in requested order
            out = []
            for vid in video_ids:
                raw = cached.get(vid)
                if raw:
                    try:
                        out.append(json.loads(raw))
                    except json.JSONDecodeError:
                        logging.error(f"Failed to decode JSON for video ID {vid}: {raw}")
                        continue
            logging.info(f"Returning metadata out for {len(out)} videos")
            # now rekey by video_id
            result_dict = {}
            for item in out:
                result_dict[item['video_id']] = item
            logging.info(f"Converted out to json dict: {len(result_dict)}")
            return jsonify(result_dict)

        @app.route('/metadata', methods=['POST'])
        def post_metadata():
            items = request.get_json(force=True)
            if not isinstance(items, list):
                return jsonify({'error': 'Expected a list of metadata dicts'}), 400
            added = 0
            for item in items:
                vid = item.get('video_id')
                if vid:
                    self.cache.hset('metadata', vid, json.dumps(item))
                    added += 1
            logger.info(f"POST /metadata added {len(added)} items")
            logger.info(f"Updated length of post_metadata: {len(self.cache.hset('metadata', vid, json.dumps(item)))}")
            return jsonify({'status': 'ok', 'added': added})

    @staticmethod
    def extract_client(video_ids, service_url='http://localhost:6000'):
        """Client helper to call GET /metadata."""
        if not video_ids:
            return []
        ids_str = ','.join(video_ids)
        resp = requests.get(f"{service_url}/metadata", params={'ids': ids_str})
        resp.raise_for_status()
        return resp.json()

def make_redis_client():
    """
    Parse REDIS_URL and return a redis.Redis instance,
    including authentication if provided.
    """
    parsed = urlparse('redis://localhost:6379/0')
    kwargs = {
        'host': parsed.hostname,
        'port': parsed.port or 6379,
        'decode_responses': True
    }
    if parsed.password:
        kwargs['password'] = parsed.password
    return redis.Redis(**kwargs)

# If run standalone:
if __name__ == '__main__':
    from urllib.parse import urlparse
    r = make_redis_client()
    from flask import Flask
    app = Flask(__name__)
    svc = MetadataExtractor()
    svc.register(app)
    app.run(host='localhost', port=6000)
