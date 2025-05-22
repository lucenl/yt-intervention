# metadata_service.py
import os
import json
import logging
import requests
import redis
from flask import Flask, request, jsonify

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
            print("vid: ", vid)
            snippet = item.get('snippet', {})
            data = {
                'video_id': vid,
                'title': snippet.get('title', ''),
                'description': snippet.get('description', '')
            }
            results.append(data)
        return results

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
            missing = [vid for vid in video_ids if vid not in cached]
            print("Missing:", missing)

            if missing:
                print(f"Fetching metadata for missing IDs: {missing}")
                try:
                    new_items = self._fetch_metadata(missing)
                    for item in new_items:
                        self.cache.hset('metadata', item['video_id'], json.dumps(item))
                    # Update local copy
                    for item in new_items:
                        cached[item['video_id']] = json.dumps(item)
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
                        continue

            # now rekey by video_id
            result_dict = {}
            for item in out:
                result_dict[item['video_id']] = item
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

# Example usage in monitor.py:
# from metadata_service import MetadataService
# from flask import Flask
# app = Flask(__name__)
# svc = MetadataService()
# svc.register(app)

# If run standalone:
if __name__ == '__main__':
    from urllib.parse import urlparse
    r = make_redis_client()
    from flask import Flask
    app = Flask(__name__)
    svc = MetadataExtractor()
    svc.register(app)
    app.run(host='localhost', port=6000)
