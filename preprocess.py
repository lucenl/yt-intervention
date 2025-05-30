import os
import json
import logging
import numpy as np
import random
from pathlib import Path
from metadata_extractor import MetadataExtractor
# from roberta_classifier import RoBERTaClassifier, MulticlassClassifier
import pandas as pd
import multiprocessing as mp
import requests

# Directory name constants (must match orchestrator, sockpuppet, and monitor)
OUTPUT_DIR_NAME = "debug-output"
LOGS_DIR_NAME = "logs"
SHARED_DIR_NAME = "debug-shared"
PROFILES_DIR_NAME = "profiles"
PUPPETS_DIR_NAME = "puppets"

# Configuration that can be set via environment variables
BASE_OUTPUT_DIR = os.getenv('BASE_OUTPUT_DIR', 'experiments')
PREPROCESS_LOG_DIR = os.getenv('PREPROCESS_LOG_DIR', 'preprocess-logs')

# Ensure preprocess log directory exists
os.makedirs(PREPROCESS_LOG_DIR, exist_ok=True)

# Global logger to be configured per puppet
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a',
    force=True
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)

_file_handler = None  # Will be set per puppet

metadata_extractor = MetadataExtractor(redis_url='redis://localhost:6379/0')

class DirectoryResolver:
    """Resolves directory paths for different stages and puppets"""
    
    def __init__(self, base_output_dir: str):
        self.base_output_dir = Path(base_output_dir).resolve()
    
    def get_stage_from_puppet_id(self, puppet_id: str) -> str:
        """Determine stage from puppet ID"""
        # Intervention puppets have focus and intervention type suffixes
        intervention_suffixes = [
            "_homepage_none", "_homepage_downrank", "_homepage_replace",
            "_upnext_none", "_upnext_downrank", "_upnext_replace"
        ]
        
        for suffix in intervention_suffixes:
            if puppet_id.endswith(suffix):
                return "intervention"
        
        return "train"
    
    def get_base_puppet_id(self, puppet_id: str) -> str:
        """Extract base puppet ID by removing intervention suffixes"""
        intervention_suffixes = [
            "_homepage_none", "_homepage_downrank", "_homepage_replace",
            "_upnext_none", "_upnext_downrank", "_upnext_replace"
        ]
        
        base_id = puppet_id
        for suffix in intervention_suffixes:
            if base_id.endswith(suffix):
                base_id = base_id[:-len(suffix)]
                break
        
        return base_id
    
    def get_stage_dirs(self, stage: str) -> dict:
        """Get directory paths for a specific stage"""
        stage_base = self.base_output_dir / stage
        return {
            "output": str(stage_base / OUTPUT_DIR_NAME),
            "logs": str(stage_base / LOGS_DIR_NAME),
            "shared": str(stage_base / SHARED_DIR_NAME)
        }
    
    def get_puppet_shared_dir(self, puppet_id: str) -> str:
        """Get shared directory for a specific puppet"""
        stage = self.get_stage_from_puppet_id(puppet_id)
        stage_dirs = self.get_stage_dirs(stage)
        return os.path.join(stage_dirs["shared"], puppet_id)
    
    def get_puppet_state_file(self, puppet_id: str) -> str:
        """Get puppet state file path"""
        stage = self.get_stage_from_puppet_id(puppet_id)
        stage_dirs = self.get_stage_dirs(stage)
        
        if stage == "intervention":
            # For intervention puppets, look for the base puppet in training stage first
            base_puppet_id = self.get_base_puppet_id(puppet_id)
            train_dirs = self.get_stage_dirs("train")
            train_state_file = os.path.join(train_dirs["output"], PUPPETS_DIR_NAME, f"{base_puppet_id}.json")
            if os.path.exists(train_state_file):
                return train_state_file
        
        # Default to current stage
        return os.path.join(stage_dirs["output"], PUPPETS_DIR_NAME, f"{puppet_id}.json")
    
    def get_experiment_data_dir(self, puppet_id: str) -> str:
        """Get experiment data directory for a puppet"""
        stage = self.get_stage_from_puppet_id(puppet_id)
        stage_dirs = self.get_stage_dirs(stage)
        experiment_dir = os.path.join(stage_dirs["output"], "experiment_data", puppet_id)
        os.makedirs(experiment_dir, exist_ok=True)
        return experiment_dir

# Initialize directory resolver
dir_resolver = DirectoryResolver(BASE_OUTPUT_DIR)

def _binary_worker(metadata):
    """Send metadata to classify_roberta endpoint and return harm scores."""
    req = requests.post('http://localhost:9005/classify_roberta', json={"metadata": metadata})
    out = pd.DataFrame(req.json())
    return out["harm_score"].tolist()

def _multi_worker(metadata):
    """Send metadata to classify_multiclass endpoint and return categories."""
    req = requests.post('http://localhost:9005/classify_multiclass', json={"metadata": metadata})
    out = pd.DataFrame(req.json())
    return out["category"].tolist()

def load_or_initialize_harmless_pool(puppet_id):
    """Load existing harmless pool or initialize with 10 non-overlapping videos."""
    experiment_data_dir = dir_resolver.get_experiment_data_dir(puppet_id)
    pool_file = os.path.join(experiment_data_dir, "harmless_pool.json")
    
    harmless_pool = []
    puppet_state_file = dir_resolver.get_puppet_state_file(puppet_id)
    
    if os.path.exists(pool_file):
        with open(pool_file, "r") as f:
            harmless_pool = json.load(f)
        logging.info(f"Loaded existing harmless pool from {pool_file}")
    else:
        training_videos = set()
        if os.path.exists(puppet_state_file):
            with open(puppet_state_file, "r") as f:
                puppet_state = json.load(f)
                training_videos = set(puppet_state.get("args", {}).get("training", []))
            logging.info(f"Found {len(training_videos)} training videos from puppet state")

        # Look for harmless videos file relative to base directory
        harmless_file = os.path.join(BASE_OUTPUT_DIR, "..", "data", "training", "non_harmful.csv")
        if not os.path.exists(harmless_file):
            # Fallback locations
            fallback_locations = [
                "data/training/non_harmful.csv",
                "../data/training/non_harmful.csv",
                "./data/training/non_harmful.csv"
            ]
            for location in fallback_locations:
                if os.path.exists(location):
                    harmless_file = location
                    break
        
        if os.path.exists(harmless_file):
            logging.info(f"Loading harmless videos from {harmless_file}")
            try:
                harmless_df = pd.read_csv(harmless_file)
                if 'videoId' in harmless_df.columns:
                    all_harmless = harmless_df['videoId'].tolist()
                else:
                    # Fallback: assume first column or single column
                    all_harmless = harmless_df.iloc[:, 0].tolist()
                
                available_harmless = [vid for vid in all_harmless if vid not in training_videos]
                harmless_pool = random.sample(available_harmless, min(10, len(available_harmless)))
                logging.info(f"Initialized harmless pool with {len(harmless_pool)} videos from {len(available_harmless)} available")
            except Exception as e:
                logging.error(f"Error reading harmless videos file {harmless_file}: {e}")
                harmless_pool = []
        else:
            logging.warning(f"Harmless videos file not found at {harmless_file}")
            harmless_pool = []
        
        # Save initialized pool
        with open(pool_file, "w") as f:
            json.dump(harmless_pool, f)
        logging.info(f"Saved harmless pool to {pool_file}")
    
    logging.info(f"Loaded/Initialized harmless pool for {puppet_id} with {len(harmless_pool)} videos")
    return harmless_pool

def save_harmless_pool(puppet_id, harmless_pool):
    """Save the updated harmless pool to a file."""
    experiment_data_dir = dir_resolver.get_experiment_data_dir(puppet_id)
    pool_file = os.path.join(experiment_data_dir, "harmless_pool.json")
    with open(pool_file, "w") as f:
        json.dump(harmless_pool, f)
    logging.info(f"Saved harmless pool for {puppet_id} with {len(harmless_pool)} videos")

def extract_metadata(video_ids, puppet_id):
    """Extract metadata for a batch of video IDs."""
    metadata = metadata_extractor.extract_client(video_ids)
    logging.info(f"Extracted metadata for {len(metadata)} videos")
    return metadata

def classify_videos(metadata, video_ids):
    redis_client = metadata_extractor.get_redis_client()
    cached_scores = redis_client.hgetall('harm_scores')
    logging.info(f"Cached harm_scores keys: {len(list(cached_scores.keys()))}...")
    
    harm_scores = {}
    uncached_metadata = []
    
    for video_id in video_ids:
        if video_id not in metadata:
            logging.error(f"Metadata missing for video_id: {video_id}")
            raise ValueError(f"Metadata missing for video_id: {video_id}")
        item = metadata[video_id]
        if not isinstance(item, dict) or 'video_id' not in item:
            logging.error(f"Invalid metadata for video_id {video_id}: {item}")
            raise ValueError(f"Invalid metadata for video_id {video_id}")
        
        if video_id in cached_scores:
            try:
                score = float(cached_scores[video_id])
                harm_scores[video_id] = score
                logging.debug(f"Retrieved cached score for {video_id}: {score}")
            except ValueError:
                logging.warning(f"Invalid harm score for {video_id}, reclassifying")
                uncached_metadata.append(item)
        else:
            uncached_metadata.append(item)
    
    if uncached_metadata:
        logging.info(f"Classifying {len(uncached_metadata)} uncached videos")
        new_scores = _binary_worker(uncached_metadata)
        
        for item, score in zip(uncached_metadata, new_scores):
            video_id = item['video_id']
            harm_scores[video_id] = score
            try:
                redis_client.hset('harm_scores', video_id, str(score))
                logging.debug(f"Cached score for {video_id}: {score}")
            except Exception as e:
                logging.error(f"Failed to cache score for {video_id}: {e}")
        logging.info(f"Cached {len(new_scores)} new harm scores")
    
    harm_scores_list = [harm_scores[vid] for vid in video_ids]
    logging.info(f"Final harm_scores: {harm_scores_list}")
    return harm_scores_list

def categorize_harmful_videos(metadata, harm_scores, harm_threshold, video_ids):
    redis_client = metadata_extractor.get_redis_client()
    cached_categories = redis_client.hgetall('categories')
    logging.info(f"Cached categories keys: {len(list(cached_categories.keys()))} entries...")
    
    categories = [""] * len(harm_scores)
    harmful_indices = [i for i, s in enumerate(harm_scores) if s > harm_threshold]
    
    uncached_harmful_md = []
    uncached_harmful_indices = []
    
    for idx in harmful_indices:
        video_id = video_ids[idx]
        if not isinstance(metadata[video_id], dict) or 'video_id' not in metadata[video_id]:
            logging.error(f"Invalid metadata for video_id {video_id}: {metadata[video_id]}")
            raise ValueError(f"Invalid metadata for video_id {video_id}")
        if video_id in cached_categories:
            categories[idx] = cached_categories[video_id].decode('utf-8') if isinstance(cached_categories[video_id], bytes) else cached_categories[video_id]
            logging.debug(f"Retrieved cached category for {video_id}: {categories[idx]}")
        else:
            uncached_harmful_md.append(metadata[video_id])
            uncached_harmful_indices.append(idx)
    
    if uncached_harmful_md:
        logging.info(f"Categorizing {len(uncached_harmful_md)} uncached harmful videos")
        new_cats = _multi_worker(uncached_harmful_md)
        
        for idx, cat, item in zip(uncached_harmful_indices, new_cats, uncached_harmful_md):
            categories[idx] = cat
            try:
                redis_client.hset('categories', item['video_id'], cat)
                logging.debug(f"Cached category for {item['video_id']}: {cat}")
            except Exception as e:
                logging.error(f"Failed to cache category for {item['video_id']}: {e}")
        logging.info(f"Cached {len(new_cats)} new categories")
    
    logging.info(f"Assigned categories: {categories}")
    return categories, harmful_indices

def apply_downrank_intervention(video_ids, harm_scores):
    """Apply downrank intervention by sorting videos by harm score."""
    sorted_pairs = sorted(zip(video_ids, harm_scores), key=lambda pair: pair[1])
    modified_video_ids, aligned_scores = map(list, zip(*sorted_pairs))
    logging.info(f"Reranked videos by harm score: {modified_video_ids}")
    return modified_video_ids, aligned_scores

def apply_replace_intervention(video_ids, harm_scores, harmful_indices, harmless_pool, harm_threshold):
    """Apply replace intervention by substituting harmful videos with harmless ones."""
    modified_video_ids = video_ids.copy()
    # make a local copy so removals don't affect the original pool
    temp_pool = harmless_pool.copy()
    used_videos = set()

    for i in harmful_indices:
        if not temp_pool:
            logging.warning(f"No harmless videos left in pool for replacement at index {i}, skipping")
            break
        replacement = random.choice(temp_pool)
        temp_pool.remove(replacement)
        modified_video_ids[i] = replacement
        used_videos.add(replacement)
        logging.info(f"Replaced harmful video {video_ids[i]} at index {i} with {replacement}, removed from pool")
    
    # Refill pool with newly discovered non-harmful videos if needed
    non_harmful_indices = [i for i, score in enumerate(harm_scores) if score <= harm_threshold and video_ids[i] not in harmless_pool]
    for idx in non_harmful_indices:
        harmless_pool.append(video_ids[idx])
    logging.info(f"Refilled harmless pool with {len(non_harmful_indices)} new non-harmful videos. New pool size: {len(harmless_pool)}")

    aligned_scores = [0] * len(video_ids)  # Focus on position for decay-weighted random
    logging.info(f"Videos after replacement: {modified_video_ids}")
    logging.info(f"Updated harmless pool size: {len(harmless_pool)}")
    return modified_video_ids, aligned_scores, harmless_pool

def apply_no_intervention(video_ids):
    """Apply no intervention, keeping original video IDs."""
    logging.info("No intervention applied")
    return video_ids.copy(), [0] * len(video_ids)

def select_video_decay_weighted(modified_video_ids, aligned_scores):
    """Select a video using decay-weighted random based on position and scores."""
    decay_factor = 0.9
    weights = [(1.0 - score) * (decay_factor ** i) for i, score in enumerate(aligned_scores)]
    weights_sum = sum(weights)
    normalized_weights = [w / weights_sum if weights_sum > 0 else 1.0 / len(modified_video_ids) for w in weights]
    selected_index = np.random.choice(len(modified_video_ids), p=normalized_weights)
    selected_video_id = modified_video_ids[selected_index]
    logging.info(f"Selected video (decay-weighted): {selected_video_id} at index {selected_index} with weights {normalized_weights}")
    return selected_video_id, selected_index

def save_experiment_data(puppet_id, round_num, video_ids, harm_scores, harm_threshold, categories, modified_video_ids, selected_video_id):
    """Save experiment data to a JSON file."""
    experiment_data_dir = dir_resolver.get_experiment_data_dir(puppet_id)
    experiment_file = os.path.join(experiment_data_dir, f"round_{round_num}.json")
    
    num_harmful = sum(1 for score in harm_scores if score > harm_threshold)
    experiment_data = {
        "round": round_num,
        "recommendations": video_ids,
        "harm_scores": harm_scores,
        "categories": categories,
        "modified_recommendations": modified_video_ids,
        "selected_video": selected_video_id,
        "num_harmful_videos": num_harmful,
        "harm_category_counts": {cat: categories.count(cat) for cat in ['HH', 'SXL', 'PH']},
    }
    with open(experiment_file, "w") as f:
        json.dump(experiment_data, f, indent=4)
    logging.info(f"Saved experiment data to {experiment_file}")

def save_next_video(puppet_shared_dir, round_num, selected_video_id):
    """Save the selected video ID for the next round."""
    next_video_file = os.path.join(puppet_shared_dir, f"next_video_{round_num}.txt")
    logging.info(f"Saving next video ID to {next_video_file} for round {round_num}")
    
    # Ensure directory exists
    os.makedirs(puppet_shared_dir, exist_ok=True)
    
    with open(next_video_file, "w") as f:
        f.write(selected_video_id)
    logging.info(f"Saved next video ID to {next_video_file}")

def signal_completion(puppet_shared_dir, round_num):
    """Signal completion of the current round."""
    done_file = os.path.join(puppet_shared_dir, f"done_{round_num}.txt")
    
    # Ensure directory exists
    os.makedirs(puppet_shared_dir, exist_ok=True)
    
    with open(done_file, "w") as f:
        f.write("Done")
    logging.info(f"Signaled completion with {done_file}")

def preprocess(puppet_id, round_num, intervention_type, focus="homepage", harm_threshold=0.8):
    """Main preprocessing function coordinating the pipeline."""
    global _file_handler
    
    # Setup logging for this puppet
    if _file_handler:
        logging.getLogger().removeHandler(_file_handler)
    
    log_file = os.path.join(PREPROCESS_LOG_DIR, f"{puppet_id}_preprocess.log")
    _file_handler = logging.FileHandler(log_file, mode='a')
    _file_handler.setLevel(logging.INFO)
    _file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(_file_handler)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting preprocessing for puppet {puppet_id}, round {round_num}, intervention {intervention_type}, focus {focus}")
    logger.info(f"Base output directory: {BASE_OUTPUT_DIR}")
    
    try:
        # Get puppet shared directory
        puppet_shared_dir = dir_resolver.get_puppet_shared_dir(puppet_id)
        logger.info(f"Puppet shared directory: {puppet_shared_dir}")
        
        # Get recommendations from the previous round
        recommendations_file = os.path.join(puppet_shared_dir, f"recommendations_{round_num}.txt")
        if not os.path.exists(recommendations_file):
            raise FileNotFoundError(f"Recommendations file not found: {recommendations_file}")
        
        with open(recommendations_file, "r") as f:
            video_ids = [line.strip() for line in f.readlines() if line.strip()]
        
        if not video_ids:
            raise ValueError(f"No video IDs found in recommendations file: {recommendations_file}")
        
        logger.info(f"Loaded {len(video_ids)} recommendation IDs for round {round_num}")
        
        # Extract and classify data
        metadata = extract_metadata(video_ids, puppet_id)
        logger.info(f"Extracted metadata for round {round_num}")
        
        harm_scores = classify_videos(metadata, video_ids)
        categories, harmful_indices = categorize_harmful_videos(metadata, harm_scores, harm_threshold, video_ids)
        logger.info(f"Classified videos for round {round_num}")
        
        # Apply intervention
        if intervention_type == "downrank":
            modified_video_ids, aligned_scores = apply_downrank_intervention(video_ids, harm_scores)
        elif intervention_type == "replace":
            # Load or initialize harmless pool
            harmless_pool = load_or_initialize_harmless_pool(puppet_id)
            modified_video_ids, aligned_scores, updated_pool = apply_replace_intervention(
                video_ids, harm_scores, harmful_indices, harmless_pool, harm_threshold
            )
            save_harmless_pool(puppet_id, updated_pool)  # Persist updated pool
        elif intervention_type == "none":
            modified_video_ids, aligned_scores = apply_no_intervention(video_ids)
        else:
            raise ValueError(f"Unknown intervention type: {intervention_type}")
        
        logger.info(f"Applied {intervention_type} intervention for round {round_num}")

        # Select video
        selected_video_id, _ = select_video_decay_weighted(modified_video_ids, aligned_scores)
        logger.info(f"Selected video for round {round_num}: {selected_video_id}")
        
        # Save results
        save_experiment_data(puppet_id, round_num, video_ids, harm_scores, harm_threshold, 
                           categories, modified_video_ids, selected_video_id)
        logger.info(f"Saved experiment results for round {round_num}")
        
        save_next_video(puppet_shared_dir, round_num, selected_video_id)
        logger.info(f"Saved next video for round {round_num}")
        
        signal_completion(puppet_shared_dir, round_num)
        logger.info(f"Signaled completion for round {round_num}")
        
        logger.info(f"Preprocessing completed successfully for puppet {puppet_id}, round {round_num}")
        
    except Exception as e:
        logger.error(f"Error during preprocessing for puppet {puppet_id}, round {round_num}: {e}")
        logger.error(f"Exception details:", exc_info=True)
        raise

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python preprocess.py <puppet_id> <round_num> <intervention_type> [focus]")
        sys.exit(1)
    
    puppet_id = sys.argv[1]
    round_num = int(sys.argv[2])
    intervention_type = sys.argv[3]
    focus = sys.argv[4] if len(sys.argv) > 4 else "homepage"
    
    try:
        preprocess(puppet_id, round_num, intervention_type, focus)
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        sys.exit(1)
