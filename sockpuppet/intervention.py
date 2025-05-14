"""
YouTube Recommendation Intervention Module

This module implements intervention strategies for YouTube recommendations
to reduce harmful content exposure. It logs all experiment data, including
recommendations, predictions, and metadata, into a unified experiment_log.json file.
"""

from ytdriver import YTDriver, Video, VideoUnavailableException
import logging
import json
import time
import os
import random
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from metadata_extractor import MetadataExtractor
from roberta_classifier import RoBERTaClassifier
import traceback
import tempfile

def run_intervention(args, puppet=None, logger=None, initial_upnext=None, initial_homepage=None, focus="homepage"):
    """
    Run intervention experiment with the following steps per round:
    1. Get initial recommendations from training end (first round) or based on focus (subsequent rounds).
    2. Load data from the previous round's experiment log (or round 0 for the first round).
    3. Use the pre-sampled harmless reservoir from puppet state or args.
    4. Classify recommendations using RoBERTa classifier.
    5. Rerank recommendations using the specified intervention strategy.
    6. Select a video using the specified selection strategy.
    7. Watch the selected video.
    8. Fetch new recommendations based on focus, extract metadata, and save to experiment log.
    Repeat for the specified number of rounds.

    Args:
        args: Arguments dictionary including model path, rounds, etc.
        puppet: Optional puppet object if called from sockpuppet.py
        logger: Optional logger
        initial_upnext: List of up-next recommendation video IDs from training (optional)
        initial_homepage: List of homepage recommendation video IDs from training (optional)
        focus: String specifying focus of recommendations ("homepage" or "up-next")

    Returns:
        Puppet object with experiment results
    """
    # Setup logging (using puppetId, step, intervention type, and timestamp)
    if logger is None:
        intervention_type = args.get("intervention_type", "unknown")
        log_filename = f"{args['puppetId']}_intervention_{intervention_type}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=os.path.join('/logs', log_filename),
            filemode='a'
        )
        logger = logging.getLogger(__name__)
    
    logger.info(f"Entering run_intervention with args: {args}, puppet: {puppet['puppetId']}")
    
    # Extract parameters
    puppet_id = args["puppetId"]
    output_dir = args.get("outputDir", "/output")
    model_path = args.get("model_path", "/app/models/roberta_checkpoint")
    intervention_type = args.get("intervention_type", "none")
    selection_type = args.get("selection_type", "decay_weighted_random")
    num_rounds = int(args.get("rounds", 10))
    watch_duration = int(args.get("duration", 30))
    harm_threshold = float(args.get("harm_threshold", 0.8))
    
    # Get harmless reservoir from puppet state if available, otherwise from args
    harmless_reservoir_ids = []
    if puppet and "args" in puppet and "harmless_reservoir" in puppet["args"]:
        harmless_reservoir_ids = puppet["args"]["harmless_reservoir"]
    else:
        harmless_reservoir_ids = args.get("harmless_reservoir", [])
    
    # Setup directories for experiment data
    experiment_dir = os.path.join(output_dir, puppet_id)
    os.makedirs(experiment_dir, exist_ok=True)
    metadata_dir = os.path.join(experiment_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    
    # Initialize metadata extractor with specified timeout and workers
    metadata_extractor = MetadataExtractor(
        output_dir=metadata_dir,
        timeout=60,
        max_workers=4
    )
    
    # Initialize classifier if model path provided
    classifier = None
    if model_path:
        try:
            classifier = RoBERTaClassifier(
                model_path=model_path,
                threshold=harm_threshold,
                logger=logger
            )
        except Exception as e:
            logger.error(f"Error initializing classifier: {e}")
            raise
    
    # Initialize puppet if not provided
    if puppet is None:
        profile_dir = args.get("profile_dir")
        if not profile_dir:
            raise KeyError("profile_dir is required in args when puppet is not provided")
        puppet = {
            "puppetId": puppet_id,
            "driver": YTDriver(profile_dir=profile_dir, use_virtual_display=True),
            "start_time": datetime.now(),
            "actions": [],
            "rounds": [],
            "harmful_exposure": []
        }
    
    try:
        # Step 3: Use the pre-sampled harmless reservoir
        harmless_reservoir = []
        if harmless_reservoir_ids:
            harmless_reservoir = [Video(None, f"https://youtube.com/watch?v={vid}") for vid in harmless_reservoir_ids]
            logger.info(f"Using pre-sampled harmless reservoir with {len(harmless_reservoir)} videos")
        else:
            logger.warning("No pre-sampled harmless reservoir provided. Reservoir will be empty.")
        
        # Add start action to puppet
        add_action(puppet, "intervention_start")
        
        # Step 1: Set initial recommendations based on focus
        initial_recommendations = None
        if focus == "homepage" and initial_homepage:
            initial_recommendations = [Video(None, f"https://youtube.com/watch?v={vid}") for vid in initial_homepage]
            add_action(puppet, "using_initial_homepage_recommendations", initial_homepage)
        elif focus == "up-next" and initial_upnext:
            initial_recommendations = [Video(None, f"https://youtube.com/watch?v={vid}") for vid in initial_upnext]
            add_action(puppet, "using_initial_upnext_recommendations", initial_upnext)
        else:
            logger.warning(f"No valid initial recommendations for focus: {focus}")
            return puppet
        
        # Run intervention rounds
        recommendations = initial_recommendations
        for round_num in range(1, num_rounds + 1):
            logger.info(f"Starting round {round_num}")
            
            # Step 2: Load data from the previous round (or round 0 for round 1)
            prev_round = 0 if round_num == 1 else round_num - 1
            log_file = os.path.join(metadata_dir, "experiment_log.json")
            data = {"rounds": []}
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        logger.warning(f"Corrupted experiment log {log_file}, starting fresh")
            
            prev_round_data = next((r for r in data["rounds"] if r["round_number"] == prev_round and r["focus"] == focus), None)
            if not prev_round_data:
                logger.error(f"No log data found for round {prev_round} with focus {focus}")
                raise ValueError(f"No log data for round {prev_round}")
            video_ids = prev_round_data["recommendations"]
            recommendations = [Video(None, f"https://youtube.com/watch?v={vid}") for vid in video_ids]
            logger.info(f"Recommendations for round {round_num}: {video_ids}")
            
            # Step 4: Classify recommendations
            harmful_pct = 0
            harm_scores = [0.0] * len(recommendations)
            harmful_count = 0
            if classifier:
                metadata_entries = prev_round_data["metadata"]
                metadata_df = pd.DataFrame(metadata_entries)
                # Binary classification
                classified_df = classifier.classify_batch(metadata_df)
                harm_scores = [classified_df[classified_df['video_id'] == vid]['harm_score'].iloc[0] for vid in video_ids]
                harmful_count = sum(1 for score in harm_scores if score > harm_threshold)
                harmful_pct = (harmful_count / len(recommendations)) * 100 if recommendations else 0
                source = "Homepage" if focus == "homepage" else "Up-Next"
                logger.info(f"{source} harmful percentage: {harmful_pct:.2f}%")
                logger.info(f"{source} harmful count: {harmful_count}")
                # Multiclass classification
                harmful_indices = [i for i, score in enumerate(harm_scores) if score > harm_threshold]
                if harmful_indices:
                    harmful_videos_df = metadata_df[metadata_df['video_id'].isin([video_ids[i] for i in harmful_indices])]
                    multiclass_results = multiclass_classifier.classify_batch(harmful_videos_df)
                    for idx, video_id in enumerate(video_ids):
                        if idx in harmful_indices:
                            category = multiclass_results[multiclass_results['video_id'] == video_id]['category'].iloc[0]
                            category_labels[idx] = category
                            logger.info(f"Video {video_id} classified as harmful with category: {category}")
            
                
            
            # Step 5: Rerank using intervention strategy
            modified_recommendations = recommendations.copy()
            if intervention_type == 'downrank':
                sorted_pairs = sorted(zip(recommendations, harm_scores), key=lambda pair: pair[1])  # Sort by harm score (ascending)
                modified_recommendations, aligned_scores = map(list, zip(*sorted_pairs))
                logger.info(f"Reranked recommendations: {[vid.videoId for vid in modified_recommendations]}")
                logger.info(f"Aligned harm scores: {aligned_scores}")\
                    
            elif intervention_type == 'replace' and harmless_reservoir:
                # Replace harmful videos with ones from the reservoir
                current_video_ids = set(video.videoId for video in modified_recommendations)
                used_replacements = set()
                for i, (video, score) in enumerate(zip(modified_recommendations, harm_scores)):
                    if score > harm_threshold:  # Harmful video
                        available_replacements = [v for v in harmless_reservoir if v.videoId not in current_video_ids and v.videoId not in used_replacements]
                        if available_replacements:
                            replacement = random.choice(available_replacements)
                            modified_recommendations[i] = replacement
                            used_replacements.add(replacement.videoId)
                            harmless_reservoir.remove(replacement)
                            logger.info(f"Replaced harmful video {video.videoId} (score: {score}) with {replacement.videoId}")
                        else:
                            logger.warning(f"No suitable replacement found for harmful video {video.videoId} (score: {score})")
                logger.info(f"Modified recommendations: {[vid.videoId for vid in modified_recommendations]}")
    
            # Step 6: Select video to watch
            if not modified_recommendations:
                logger.warning("No videos available to select. Terminating intervention.")
                puppet["harmful_exposure"].append({
                    "round": round_num,
                    "harmful_count": 0,
                    "harmful_percentage": 0.0
                })
                raise ValueError("No videos available to select")
                
            if selection_type == 'decay_weighted_random':
                decay_factor = 0.9
                weights = [(1.0 - score) * (decay_factor ** i) for i, score in enumerate(aligned_scores)]
                weights_sum = sum(weights)
                if weights_sum > 0:
                    normalized_weights = weights / weights_sum
                else:
                    normalized_weights = np.ones_like(weights) / len(weights)
                    
                # guard against tiny floating‐point drift
                normalized_weights = np.clip(normalized_weights, 0.0, 1.0)
                normalized_weights /= normalized_weights.sum()

                selected_index = np.random.choice(len(modified_recommendations), p=normalized_weights)
                selected_video = modified_recommendations[selected_index]
                logger.info(f"Decay-adjusted weights: {normalized_weights}")
                logger.info(f"Selected video index: {selected_index} with weight: {normalized_weights[selected_index]}")
                logger.info(f"Selected video: {selected_video.videoId}")
                
            elif selection_type == 'random':
                selected_video = random.choice(modified_recommendations)
                logger.info(f"Selected video (random): {selected_video.videoId}")
                
            else:  # Default to top
                selected_video = modified_recommendations[0]
                logger.info(f"Selected video (top): {selected_video.videoId}")
            
            add_action(puppet, "select_video", selected_video.videoId)
            
            # Step 7: Watch the selected video
            try:
                logger.info(f"Watching video: {selected_video.videoId}")
                puppet["driver"].play(selected_video, duration=watch_duration)
                logger.info(f"Finished watching video: {selected_video.videoId}")
                add_action(puppet, "watch", selected_video.videoId)
                
                round_result = {
                    "round": round_num,
                    "timestamp": datetime.now().isoformat(),
                    "harmful_count": harmful_count,
                    "harmful_percentage": harmful_pct,
                    "selected_video": selected_video.videoId
                }
                
                # Step 8: Fetch new recommendations based on focus, extract metadata, and save to experiment log
                if round_num < num_rounds:
                    if focus == "homepage":
                        recommendations = puppet["driver"].get_homepage_recommendations(scroll_times=6)
                        logger.info(f"Got {len(recommendations)} homepage recommendations")
                        recommendations = recommendations[:25]  # Trim to 25 as in training
                        logger.info(f"Trimmed to {len(recommendations)} homepage recommendations")
                        add_action(puppet, "get_homepage_recommendations", [vid.videoId for vid in recommendations])
                        video_ids = [video.videoId for video in recommendations]
                        metadata = metadata_extractor.extract_metadata_batch(video_ids)
                        if classifier:
                            metadata_df = pd.DataFrame(metadata)
                            harm_scores = classifier.classify_batch(metadata_df)['harm_score'].tolist()
                        else:
                            harm_scores = [0.0] * len(video_ids)
                        save_experiment_log(metadata_dir, round_num, focus, video_ids, harm_scores, metadata)
                    elif focus == "up-next":
                        recommendations = puppet["driver"].get_upnext_recommendations(topn=12)
                        logger.info(f"Got {len(recommendations)} up-next recommendations")
                        add_action(puppet, "get_upnext_recommendations", [vid.videoId for vid in recommendations])
                        video_ids = [video.videoId for video in recommendations]
                        metadata = metadata_extractor.extract_metadata_batch(video_ids)
                        if classifier:
                            metadata_df = pd.DataFrame(metadata)
                            harm_scores = classifier.classify_batch(metadata_df)['harm_score'].tolist()
                        else:
                            harm_scores = [0.0] * len(video_ids)
                        save_experiment_log(metadata_dir, round_num, focus, video_ids, harm_scores, metadata)
    
            except Exception as e:
                logger.error(f"Error watching video: {str(e)}\n{traceback.format_exc()}")
                round_result = {
                    "round": round_num,
                    "timestamp": datetime.now().isoformat(),
                    "harmful_count": harmful_count,
                    "harmful_percentage": harmful_pct,
                    "error": str(e)
                }
                puppet["rounds"].append(round_result)
                puppet["harmful_exposure"].append({
                    "round": round_num,
                    "harmful_count": harmful_count,
                    "harmful_percentage": harmful_pct
                })
                raise
            
            puppet["rounds"].append(round_result)
            puppet["harmful_exposure"].append({
                "round": round_num,
                "harmful_count": harmful_count,
                "harmful_percentage": harmful_pct
            })
            
            time.sleep(2)  # Brief pause between rounds
        
        # Finalize intervention
        add_action(puppet, "intervention_end")
        generate_visualization(puppet, os.path.join(experiment_dir, "visualizations"))
        
        logger.info(f"Exiting run_intervention for puppet {puppet['puppetId']}")
        return puppet
        
    except Exception as e:
        logger.error(f"Error running intervention: {str(e)}\n{traceback.format_exc()}")
        if puppet.get("standalone", False) and "driver" in puppet:
            puppet["driver"].close()
        puppet["error"] = str(e)
        logger.info(f"Exiting run_intervention with error for puppet {puppet['puppetId']}")
        raise

def add_action(puppet, action, params=None):
    """
    Add an action to the puppet's action log.

    Args:
        puppet: Puppet object to log the action
        action: String describing the action
        params: Optional parameters for the action
    """
    puppet["actions"].append({
        "action": action,
        "params": params,
        "timestamp": datetime.now().isoformat()
    })

def save_experiment_log(metadata_dir, round_num, focus, video_ids, predictions=None, metadata_entries=None):
    """
    Save experiment data (recommendations, predictions, metadata) to a unified JSON log file.

    Args:
        metadata_dir: Directory to store the experiment log
        round_num: Round number of the experiment
        focus: Focus of recommendations ("homepage" or "up-next")
        video_ids: List of video IDs from recommendations
        predictions: List of harm scores (or None if not applicable)
        metadata_entries: List of metadata entries for the videos
    """
    log_file = os.path.join(metadata_dir, "experiment_log.json")
    data = {"rounds": []}
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                logging.warning(f"Corrupted experiment log {log_file}, starting fresh")
    
    round_data = {
        "round_number": round_num,
        "focus": focus,
        "recommendations": video_ids,
        "predictions": predictions if predictions is not None else [None] * len(video_ids),
        "metadata": metadata_entries if metadata_entries is not None else []
    }
    data["rounds"].append(round_data)
    
    # Use atomic write to prevent corruption with concurrent access
    with tempfile.NamedTemporaryFile(mode="w", dir=metadata_dir, delete=False) as tmp_file:
        json.dump(data, tmp_file, default=str, indent=4)
    os.replace(tmp_file.name, log_file)
    logging.info(f"Appended experiment log for round {round_num} to {log_file}")

def generate_visualization(puppet, output_dir):
    """
    Generate a visualization of harmful content exposure over rounds.

    Args:
        puppet: Puppet object containing harmful exposure data
        output_dir: Directory to save the visualization
    """
    os.makedirs(output_dir, exist_ok=True)
    exposure_data = puppet.get('harmful_exposure', [])
    if not exposure_data:
        return
    rounds = [d.get('round', i+1) for i, d in enumerate(exposure_data)]
    harmful_counts = [d.get('harmful_count', 0) for d in exposure_data]
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, harmful_counts, 'b-o', linewidth=2, label='Harmful Videos')
    plt.title(f"Harmful Content Exposure - {puppet['puppetId']}", fontsize=14)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Harmful Content', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{puppet['puppetId']}_exposure.png"), dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    """
    Entry point for running the intervention as a standalone script.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Run YouTube recommendation intervention")
    parser.add_argument("--puppet-id", required=True, help="Puppet ID")
    parser.add_argument("--profile-dir", required=True, help="Path to puppet profile directory")
    parser.add_argument("--output-dir", default="/app/output", help="Output directory")
    parser.add_argument("--model-path", help="Path to RoBERTa model checkpoint")
    parser.add_argument("--intervention", default="none", choices=["none", "downrank", "replace"],
                        help="Intervention strategy")
    parser.add_argument("--selection", default="decay_weighted_random", 
                        choices=["top", "random", "decay_weighted_random"],
                        help="Video selection strategy")
    parser.add_argument("--rounds", type=int, default=10, help="Number of rounds")
    parser.add_argument("--duration", type=int, default=30, help="Video watch duration (seconds)")
    parser.add_argument("--harm-threshold", type=float, default=0.8, help="Harm classification threshold")
    parser.add_argument("--focus", default="homepage", choices=["homepage", "up-next"],
                        help="Focus on homepage or up-next recommendations")
    parser.add_argument("--training", nargs="+", help="List of training video IDs")
    parser.add_argument("--harmless-reservoir", nargs="+", help="List of pre-sampled harmless video IDs")
    args = parser.parse_args()
    args_dict = {
        "puppetId": args.puppet_id,
        "profile_dir": args.profile_dir,
        "outputDir": args.output_dir,
        "model_path": args.model_path,
        "intervention_type": args.intervention,
        "selection_type": args.selection,
        "rounds": args.rounds,
        "duration": args.duration,
        "harm_threshold": args.harm_threshold,
        "focus": args.focus,
        "training": args.training,
        "harmless_reservoir": args.harmless_reservoir,
        "standalone": True
    }
    run_intervention(args_dict)