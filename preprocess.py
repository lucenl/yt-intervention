import os
import json
import logging
import numpy as np
import random
from metadata_extractor import MetadataExtractor
# from roberta_classifier import RoBERTaClassifier, MulticlassClassifier
import pandas as pd
import multiprocessing as mp
import requests

SHARED_DIR = "./shared"
LOCAL_LOG_DIR = "./local_logs"
EXPERIMENT_DATA_DIR = "./experiment_data"
os.makedirs(LOCAL_LOG_DIR, exist_ok=True)
os.makedirs(EXPERIMENT_DATA_DIR, exist_ok=True)

# worker for binary classification
def _binary_worker(args):
    batch_metadata, model_path, batch_size = args
    req = requests.post('http://localhost:9000/classify_roberta', json={"metadata": batch_metadata})
    out = pd.DataFrame(req.json())
    return out["harm_score"].tolist()

# worker for multiclass classification
def _multi_worker(args):
    batch_metadata, model_path, batch_size = args
    req = requests.post('http://localhost:9000/classify_multiclass', json={"metadata": batch_metadata})
    out = pd.DataFrame(req.json())
    return out["category"].tolist()

def load_or_initialize_harmless_pool(puppet_id):
    """Load existing harmless pool or initialize with 10 non-overlapping videos."""
    pool_file = os.path.join(EXPERIMENT_DATA_DIR, puppet_id, "harmless_pool.json")
    os.makedirs(os.path.dirname(pool_file), exist_ok=True)
    
    harmless_pool = []
    puppet_state_file = os.path.join("output", "puppets", f"{puppet_id}.json")
    
    if os.path.exists(pool_file):
        with open(pool_file, "r") as f:
            harmless_pool = json.load(f)
    else:
        training_videos = set()
        if os.path.exists(puppet_state_file):
            with open(puppet_state_file, "r") as f:
                puppet_state = json.load(f)
                training_videos = set(puppet_state["args"].get("training", []))

        harmless_file = os.path.join("data/training", "non_harmful.csv")
        if os.path.exists(harmless_file):
            logging.info(f"Loading harmless videos from {harmless_file}")
            with open(harmless_file, "r") as f:
                all_harmless = [line.strip() for line in f if line.strip()]
            available_harmless = [vid for vid in all_harmless if vid not in training_videos]
            harmless_pool = random.sample(available_harmless, min(10, len(available_harmless)))
            logging.info(f"Initialized harmless pool with {len(harmless_pool)} videos from {len(available_harmless)} available")
        with open(pool_file, "w") as f:
            json.dump(harmless_pool, f)
            logging.info(f"Saved harmless pool to {pool_file}")
    
    logging.info(f"Loaded/Initialized harmless pool for {puppet_id} with {len(harmless_pool)} videos")
    return harmless_pool

def save_harmless_pool(puppet_id, harmless_pool):
    """Save the updated harmless pool to a file."""
    pool_file = os.path.join(EXPERIMENT_DATA_DIR, puppet_id, "harmless_pool.json")
    with open(pool_file, "w") as f:
        json.dump(harmless_pool, f)
    logging.info(f"Saved harmless pool for {puppet_id} with {len(harmless_pool)} videos")

def extract_metadata(video_ids, puppet_id):
    """Extract metadata for a batch of video IDs."""
    metadata_extractor = MetadataExtractor(redis_url='redis://localhost:6379/0')
    logging.info(f"Extracting metadata Initialized")
    metadata = metadata_extractor.extract_client(video_ids)
    logging.info(f"Extracted metadata for {len(metadata)} videos")
    return metadata

def classify_videos(metadata, batch_size=2):
    """Classify videos using RoBERTa to determine harm scores."""
    # split metadata into micro-batches
    batches = [metadata[i : i + batch_size]
               for i in range(0, len(metadata), batch_size)]
    # for each, spawn a subprocess that loads its own classifier and then exits
    with mp.Pool(processes=min(len(batches), 1)) as pool:
        args = [(bat, "./models/binary", batch_size) for bat in batches]
        results = pool.map(_binary_worker, args)
    # flatten back into one list
    harm_scores = [score for sub in results for score in sub]
    logging.info(f"Classified videos with harm scores: {harm_scores}")
    return harm_scores

def categorize_harmful_videos(metadata, harm_scores, harm_threshold, batch_size=2):
    """Categorize harmful videos using a multiclass classifier."""
    harmful_indices = [i for i, s in enumerate(harm_scores) if s > harm_threshold]
    categories = [""] * len(harm_scores)
    if harmful_indices:
        # pull out only the harmful entries
        harmful_md = [metadata[i] for i in harmful_indices]
        # micro-batch them
        batches = [harmful_md[i : i + batch_size]
                   for i in range(0, len(harmful_md), batch_size)]
        with mp.Pool(processes=min(len(batches), 1)) as pool:
            args = [(bat, "./models/multiclass", batch_size) for bat in batches]
            results = pool.map(_multi_worker, args)
        cats = [c for sub in results for c in sub]
        # reinsert into the full list
        for idx, cat in zip(harmful_indices, cats):
            categories[idx] = cat

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
    # make a local copy so removals don’t affect the original pool
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
    experiment_file = os.path.join(EXPERIMENT_DATA_DIR, puppet_id, f"round_{round_num}.json")
    os.makedirs(os.path.dirname(experiment_file), exist_ok=True)
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
    next_video_file = os.path.join(puppet_shared_dir, f"next_video_{round_num}.txt")  # Changed from round_num + 1
    with open(next_video_file, "w") as f:
        f.write(selected_video_id)
    logging.info(f"Saved next video ID to {next_video_file}")

def signal_completion(puppet_shared_dir, round_num):
    """Signal completion of the current round."""
    done_file = os.path.join(puppet_shared_dir, f"done_{round_num}.txt")
    with open(done_file, "w") as f:
        f.write("Done")
    logging.info(f"Signaled completion with {done_file}")

def preprocess(puppet_id, round_num, intervention_type, harm_threshold=0.8):
    """Main preprocessing function coordinating the pipeline."""
    puppet_shared_dir = os.path.join(SHARED_DIR, puppet_id)
    with open(os.path.join(puppet_shared_dir, f"recommendations_{round_num}.txt"), "r") as f:
        video_ids = f.read().splitlines()
    # Setup logging
    log_file = os.path.join(LOCAL_LOG_DIR, f"{puppet_id}_preprocess.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='a'
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting preprocessing for puppet {puppet_id}, round {round_num}")

    # Extract and classify data
    metadata = extract_metadata(video_ids, puppet_id)
    harm_scores = classify_videos(metadata)
    categories, harmful_indices = categorize_harmful_videos(metadata, harm_scores, harm_threshold)

    # Apply intervention
    if intervention_type == "downrank":
        modified_video_ids, aligned_scores = apply_downrank_intervention(video_ids, harm_scores)
    elif intervention_type == "replace":
        # Load or initialize harmless pool
        harmless_pool = load_or_initialize_harmless_pool(puppet_id)
        modified_video_ids, aligned_scores, updated_pool = apply_replace_intervention(video_ids, harm_scores, harmful_indices, harmless_pool, harm_threshold)
        save_harmless_pool(puppet_id, updated_pool)  # Persist updated pool
    elif intervention_type == "none":  # none
        modified_video_ids, aligned_scores = apply_no_intervention(video_ids)

    # Select video
    selected_video_id, _ = select_video_decay_weighted(modified_video_ids, aligned_scores)

    # Save results
    save_experiment_data(puppet_id, round_num, video_ids, harm_scores, harm_threshold, categories, modified_video_ids, selected_video_id)
    save_next_video(puppet_shared_dir, round_num, selected_video_id)
    signal_completion(puppet_shared_dir, round_num)

if __name__ == "__main__":
    import sys
    puppet_id = sys.argv[1]
    round_num = int(sys.argv[2])
    intervention_type = sys.argv[3]
    preprocess(puppet_id, round_num, intervention_type)