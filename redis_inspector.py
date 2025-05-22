#!/usr/bin/env python3
"""
Redis Data Inspector for YouTube Metadata Cache
Usage: python redis_inspector.py [command] [options]
"""

import redis
import json
import sys
from datetime import datetime

class RedisInspector:
    def __init__(self, redis_url='redis://localhost:6379/0'):
        self.redis_url = redis_url
        self.cache = redis.Redis.from_url(redis_url, decode_responses=True)
        
    def test_connection(self):
        """Test Redis connection"""
        try:
            self.cache.ping()
            print(f"✅ Connected to Redis at {self.redis_url}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Redis: {e}")
            return False
    
    def get_stats(self):
        """Get basic statistics about the Redis data"""
        try:
            # Get data from all relevant hashes
            metadata_data = self.cache.hgetall('metadata')
            harm_scores_data = self.cache.hgetall('harm_scores')
            categories_data = self.cache.hgetall('categories')
            
            total_videos = len(metadata_data)
            has_metadata = 0
            has_harm_score = len(harm_scores_data)
            has_category = len(categories_data)
            categories = {}
            harm_scores = []
            
            # Process metadata
            for video_id, data_str in metadata_data.items():
                try:
                    data = json.loads(data_str)
                    if data.get('title') or data.get('description'):
                        has_metadata += 1
                except json.JSONDecodeError:
                    continue
            
            # Process harm scores
            for video_id, score_str in harm_scores_data.items():
                try:
                    score = float(score_str)
                    harm_scores.append(score)
                except ValueError:
                    continue
            
            # Process categories
            for video_id, category in categories_data.items():
                categories[category] = categories.get(category, 0) + 1
            
            # Print statistics
            print("\n📊 REDIS DATA STATISTICS")
            print("=" * 50)
            print(f"Total videos in metadata cache: {total_videos}")
            print(f"Videos with metadata (title/description): {has_metadata}")
            print(f"Videos with harm scores: {has_harm_score}")
            print(f"Videos with categories: {has_category}")
            
            if harm_scores:
                print(f"\n🎯 HARM SCORE ANALYSIS")
                print(f"Average harm score: {sum(harm_scores)/len(harm_scores):.3f}")
                print(f"Min harm score: {min(harm_scores):.3f}")
                print(f"Max harm score: {max(harm_scores):.3f}")
                high_harm = sum(1 for score in harm_scores if score > 0.8)
                print(f"Videos with harm score > 0.8: {high_harm}")
            
            if categories:
                print(f"\n📂 CATEGORY DISTRIBUTION")
                for cat, count in sorted(categories.items()):
                    print(f"  {cat}: {count}")
                    
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
    
    def list_videos(self, limit=10, show_full=False):
        """List videos in the cache"""
        try:
            metadata_data = self.cache.hgetall('metadata')
            harm_scores_data = self.cache.hgetall('harm_scores')
            categories_data = self.cache.hgetall('categories')
            count = 0
            
            print(f"\n📋 VIDEOS IN CACHE (showing first {limit})")
            print("=" * 80)
            
            for video_id, data_str in metadata_data.items():
                if count >= limit:
                    break
                    
                try:
                    data = json.loads(data_str)
                    print(f"\n🎬 Video ID: {video_id}")
                    
                    if show_full:
                        # Include harm score and category in full view
                        data_copy = data.copy()
                        data_copy['harm_score'] = harm_scores_data.get(video_id, 'N/A')
                        data_copy['category'] = categories_data.get(video_id, 'N/A')
                        print(json.dumps(data_copy, indent=2))
                    else:
                        print(f"  Title: {data.get('title', 'N/A')[:80]}...")
                        print(f"  Harm Score: {harm_scores_data.get(video_id, 'N/A')}")
                        print(f"  Category: {categories_data.get(video_id, 'N/A')}")
                    
                    count += 1
                except json.JSONDecodeError:
                    print(f"❌ Error parsing data for video {video_id}")
                    count += 1
                    
            if len(metadata_data) > limit:
                print(f"\n... and {len(metadata_data) - limit} more videos")
                
        except Exception as e:
            print(f"❌ Error listing videos: {e}")
    
    def search_video(self, video_id):
        """Search for a specific video"""
        try:
            data_str = self.cache.hget('metadata', video_id)
            harm_score = self.cache.hget('harm_scores', video_id)
            category = self.cache.hget('categories', video_id)
            
            if data_str:
                data = json.loads(data_str)
                print(f"\n🎬 VIDEO DETAILS: {video_id}")
                print("=" * 50)
                data_copy = data.copy()
                data_copy['harm_score'] = harm_score if harm_score else 'N/A'
                data_copy['category'] = category if category else 'N/A'
                print(json.dumps(data_copy, indent=2))
            else:
                print(f"❌ Video {video_id} not found in cache")
        except Exception as e:
            print(f"❌ Error searching for video: {e}")
    
    def clear_all_data(self, confirm=False):
        """Clear all data from Redis"""
        if not confirm:
            print("⚠️  WARNING: This will delete ALL data in the Redis cache!")
            print("Use --confirm flag to proceed: python redis_inspector.py clear --confirm")
            return
            
        try:
            keys = ['metadata', 'harm_scores', 'categories']
            deleted = 0
            for key in keys:
                if self.cache.delete(key):
                    deleted += 1
            if deleted > 0:
                print(f"✅ Successfully cleared {deleted} keys from Redis cache")
            else:
                print("ℹ️  No data found to clear")
        except Exception as e:
            print(f"❌ Error clearing data: {e}")
    
    def clear_classifications_only(self, confirm=False):
        """Clear only classification data, keep metadata"""
        if not confirm:
            print("⚠️  WARNING: This will remove all classification results but keep metadata!")
            print("Use --confirm flag to proceed")
            return
            
        try:
            keys = ['harm_scores', 'categories']
            deleted = 0
            for key in keys:
                if self.cache.delete(key):
                    deleted += 1
            if deleted > 0:
                print(f"✅ Cleared {deleted} classification keys from Redis")
            else:
                print("ℹ️  No classification data found to clear")
        except Exception as e:
            print(f"❌ Error clearing classifications: {e}")
    
    def export_data(self, filename=None):
        """Export all data to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"redis_export_{timestamp}.json"
            
        try:
            metadata_data = self.cache.hgetall('metadata')
            harm_scores_data = self.cache.hgetall('harm_scores')
            categories_data = self.cache.hgetall('categories')
            export_data = {}
            
            for video_id, data_str in metadata_data.items():
                try:
                    data = json.loads(data_str)
                    data['harm_score'] = harm_scores_data.get(video_id, None)
                    data['category'] = categories_data.get(video_id, None)
                    export_data[video_id] = data
                except json.JSONDecodeError:
                    continue
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
                
            print(f"✅ Exported {len(export_data)} videos to {filename}")
            
        except Exception as e:
            print(f"❌ Error exporting data: {e}")

def main():
    inspector = RedisInspector()
    
    if not inspector.test_connection():
        return
    
    if len(sys.argv) < 2:
        print("\n🔍 REDIS INSPECTOR COMMANDS")
        print("=" * 40)
        print("python redis_inspector.py stats          # Show statistics")
        print("python redis_inspector.py list [N]       # List first N videos (default 10)")
        print("python redis_inspector.py list-full [N]  # List videos with full details")
        print("python redis_inspector.py search VIDEO_ID # Search specific video")
        print("python redis_inspector.py export [file]  # Export all data to JSON")
        print("python redis_inspector.py clear --confirm # Clear ALL data")
        print("python redis_inspector.py clear-class --confirm # Clear only classifications")
        print("\n🚨 REDIS COMMANDS (run in terminal):")
        print("redis-cli flushdb    # Clear current database")
        print("redis-cli flushall   # Clear ALL databases")
        print("redis-cli ping       # Test connection")
        print("redis-cli info       # Show Redis info")
        inspector.get_stats()
        return
    
    command = sys.argv[1].lower()
    
    if command == 'stats':
        inspector.get_stats()
        
    elif command == 'list':
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        inspector.list_videos(limit, show_full=False)
        
    elif command == 'list-full':
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        inspector.list_videos(limit, show_full=True)
        
    elif command == 'search':
        if len(sys.argv) < 3:
            print("❌ Please provide a video ID to search")
            return
        inspector.search_video(sys.argv[2])
        
    elif command == 'export':
        filename = sys.argv[2] if len(sys.argv) > 2 else None
        inspector.export_data(filename)
        
    elif command == 'clear':
        confirm = '--confirm' in sys.argv
        inspector.clear_all_data(confirm)
        
    elif command == 'clear-class':
        confirm = '--confirm' in sys.argv
        inspector.clear_classifications_only(confirm)
        
    else:
        print(f"❌ Unknown command: {command}")
        main()

if __name__ == "__main__":
    main()