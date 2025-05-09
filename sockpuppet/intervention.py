"""
YouTube Recommendation Intervention Module

This module implements intervention strategies for YouTube recommendations
to reduce harmful content exposure.
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

def run_intervention(args, puppet=None, logger=None, initial_upnext=None, initial_homepage=None, focus="both"):
    """
    Run intervention experiment
    
    Args:
        args: Arguments dictionary including model path, rounds, etc.
        puppet: Optional puppet object if called from sockpuppet.py
        logger: Optional logger
        initial_upnext: List of upnext recommendation video IDs from training (optional)
        initial_homepage: List of homepage recommendation video IDs from training (optional)
        focus: String specifying focus of recommendations ("homepage", "up-next", or "both")
    
    Returns:
        Puppet object with experiment results
    """
    # Setup logging
    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
    
    logger.info(f"Entering run_intervention with args: {args}, puppet: {puppet}")
    
    # Extract parameters
    puppet_id = args["puppetId"]
    output_dir = args.get("outputDir", "/output")
    model_path = args.get("model_path", "/app/models/roberta_checkpoint")
    intervention_type = args.get("intervention_type", "none")
    selection_type = args.get("selection_type", "decay_weighted_random") 
    num_rounds = int(args.get("rounds", 10))
    watch_duration = int(args.get("duration", 30))
    harm_threshold = float(args.get("harm_threshold", 0.8))
    training_videos = args.get("training", [])
    
    # Setup directories
    experiment_dir = os.path.join(output_dir, puppet_id)
    os.makedirs(experiment_dir, exist_ok=True)
    metadata_dir = os.path.join(experiment_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    
    # Initialize metadata extractor
    metadata_extractor = MetadataExtractor(
        output_dir=metadata_dir,
        timeout=60,
        max_workers=10
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
        # Initialize harmless reservoir with non-harmful training videos (Approach 2)
        harmless_reservoir = []
        if classifier and training_videos:
            training_urls = [Video(None, f"https://youtube.com/watch?v={vid}") for vid in training_videos]
            training_ids = [vid.videoId for vid in training_urls]
            training_csv = metadata_extractor.extract_metadata_batch(training_ids, filename="metadata_training.csv")
            training_df = classifier.classify_from_csv(training_csv)
            training_scores = classifier.classify_video_list(training_urls, training_df)
            for video, score in zip(training_urls, training_scores):
                if score <= harm_threshold and video.videoId not in [v.videoId for v in harmless_reservoir]:
                    harmless_reservoir.append(video)
            logger.info(f"Initialized harmless reservoir with {len(harmless_reservoir)} non-harmful training videos")
        else:
            if not training_videos:
                logger.warning("No training videos provided. Harmless reservoir will start empty and grow dynamically.")
            if not classifier:
                logger.warning("No classifier available. Cannot initialize harmless reservoir with training videos.")
        
        logger.info(f"Puppet before first add_action: {puppet}")
        # Add start action
        add_action(puppet, "intervention_start")
        
        # Convert initial recommendations to Video objects if provided
        current_recommendations = None
        if focus in ["homepage", "both"] and initial_homepage:
            current_recommendations = [Video(None, f"https://youtube.com/watch?v={vid}") for vid in initial_homepage]
            add_action(puppet, "using_initial_homepage_recommendations", initial_homepage)
        elif focus in ["up-next", "both"] and initial_upnext:
            current_recommendations = [Video(None, f"https://youtube.com/watch?v={vid}") for vid in initial_upnext]
            add_action(puppet, "using_initial_upnext_recommendations", initial_upnext)
        else:
            logger.warning(f"No valid initial recommendations for focus: {focus}")
            return puppet
        
        # Run intervention rounds
        for round_num in range(1, num_rounds + 1):
            logger.info(f"Starting round {round_num}")
            
            # Get recommendations based on focus
            if round_num == 1 and current_recommendations:
                recommendations = current_recommendations
                logger.info(f"Using initial recommendations: {len(recommendations)} videos")
            else:
                try:
                    if focus == "homepage":
                        recommendations = puppet["driver"].get_homepage_recommendations(scroll_times=4)
                        logger.info(f"Got {len(recommendations)} homepage recommendations")
                        add_action(puppet, "get_homepage_recommendations", [vid.videoId for vid in recommendations])
                    elif focus == "up-next":
                        recommendations = puppet["driver"].get_upnext_recommendations(topn=12)
                        logger.info(f"Got {len(recommendations)} up-next recommendations")
                        add_action(puppet, "get_upnext_recommendations", [vid.videoId for vid in recommendations])
                    else:  # focus == "both"
                        recommendations = puppet["driver"].get_homepage_recommendations(scroll_times=4)
                        logger.info(f"Got {len(recommendations)} homepage recommendations")
                        add_action(puppet, "get_homepage_recommendations", [vid.videoId for vid in recommendations])
                except Exception as e:
                    logger.error(f"Failed to get recommendations for round {round_num}: {str(e)}")
                    puppet["harmful_exposure"].append({
                        "round": round_num,
                        "harmful_percentage": 0.0,
                        "error": str(e)
                    })
                    continue
            
            # Extract video IDs
            video_ids = [video.videoId for video in recommendations]
            logger.info(f"Original recommendations (before reranking): {video_ids}")
            
            # Get metadata and classify all recommendations
            metadata_csv = metadata_extractor.extract_metadata_batch(video_ids, filename=f"metadata_round_{round_num}.csv")
            logger.info(f"Metadata saved to: {metadata_csv}")
            harmful_pct = 0
            harm_scores = [0.0] * len(recommendations)
            if classifier:
                metadata_df = classifier.classify_from_csv(metadata_csv)
                harm_scores = classifier.classify_video_list(recommendations, metadata_df)
                if len(harm_scores) != len(recommendations):
                    logger.error(f"Mismatch in number of scores ({len(harm_scores)}) and recommendations ({len(recommendations)})")
                    raise ValueError("Mismatch in classification scores and recommendations")
                
                logger.info(f"Harm scores: {dict(zip(video_ids, harm_scores))}")
                harmful_count = sum(1 for score in harm_scores if score > harm_threshold)
                harmful_pct = (harmful_count / len(recommendations)) * 100 if recommendations else 0
                source = "Homepage" if focus == "homepage" else "Up-Next" if focus == "up-next" else "Homepage"
                logger.info(f"{source} harmful content: {harmful_pct:.2f}%")
                
                # Update harmless reservoir with non-harmful videos from this round (Approach 3)
                for video, score in zip(recommendations, harm_scores):
                    if score <= harm_threshold and video.videoId not in [v.videoId for v in harmless_reservoir]:
                        harmless_reservoir.append(video)
                logger.info(f"Harmless reservoir size after update: {len(harmless_reservoir)}")
            
            # Apply intervention
            modified_recommendations = recommendations.copy()
            if intervention_type == 'downrank':
                video_scores = list(zip(recommendations, harm_scores))
                sorted_pairs = sorted(video_scores, key=lambda x: x[1])  # Sort by harm score (ascending)
                modified_recommendations = [video for video, _ in sorted_pairs]
                logger.info(f"Reranked recommendations: {[vid.videoId for vid in modified_recommendations]}")
            
            elif intervention_type == 'replace' and harmless_reservoir:
                # Replace harmful videos in the top 25
                top_25 = modified_recommendations[:25]  # Limit to top 25 for replacement
                current_video_ids = set(video.videoId for video in modified_recommendations)  # IDs in current homepage
                used_replacements = set()
                for i, (video, score) in enumerate(zip(top_25, harm_scores[:25])):
                    if score > harm_threshold:  # Harmful video
                        # Filter harmless_reservoir to exclude videos already in the current recommendations
                        available_replacements = [v for v in harmless_reservoir if v.videoId not in current_video_ids and v.videoId not in used_replacements]
                        if available_replacements:
                            replacement = random.choice(available_replacements)
                            modified_recommendations[i] = replacement
                            used_replacements.add(replacement.videoId)
                            harmless_reservoir.remove(replacement)  # Remove used video
                            logger.info(f"Replaced harmful video {video.videoId} (score: {score}) with {replacement.videoId}")
                        else:
                            logger.warning(f"No suitable replacement found for harmful video {video.videoId} (score: {score})")
                logger.info(f"Modified recommendations: {[vid.videoId for vid in modified_recommendations]}")
            
            # Select video to watch
            if not modified_recommendations:
                logger.warning("No videos available to select. Skipping round.")
                puppet["harmful_exposure"].append({
                    "round": round_num,
                    "harmful_percentage": 0.0
                })
                continue
                
            if selection_type == 'decay_weighted_random':
                decay_factor = 0.9
                weights = [(1.0 - score) * (decay_factor ** i) for i, score in enumerate(harm_scores)]
                weights_sum = sum(weights)
                if weights_sum > 0:
                    normalized_weights = [w / weights_sum for w in weights]
                else:
                    normalized_weights = [1.0 / len(modified_recommendations)] * len(modified_recommendations)
                selected_index = np.random.choice(len(modified_recommendations), p=normalized_weights)
                selected_video = modified_recommendations[selected_index]
                logger.info(f"Decay-adjusted weights: {normalized_weights}")
                logger.info(f"Selected video index: {selected_index} with weight: {normalized_weights[selected_index]}")
                logger.info(f"Selected video: {selected_video.videoId}")
                
            elif selection_type == 'random':
                selected_video = random.choice(modified_recommendations)
                logger.info(f"Selected video (random): {selected_video.videoId}")
                
            else:  # top or default
                selected_video = modified_recommendations[0]
                logger.info(f"Selected video (top): {selected_video.videoId}")
            
            add_action(puppet, "select_video", selected_video.videoId)
            
            try:
                logger.info(f"Watching video: {selected_video.videoId}")
                puppet["driver"].play(selected_video, duration=watch_duration)
                logger.info(f"Finished watching video: {selected_video.videoId}")
                add_action(puppet, "watch", selected_video.videoId)
                
                sidebar_harmful_pct = 0
                if focus in ["up-next", "both"]:
                    sidebar = puppet["driver"].get_upnext_recommendations(topn=12)
                    logger.info(f"Got {len(sidebar)} sidebar recommendations")
                    add_action(puppet, "get_upnext_recommendations", [vid.videoId for vid in sidebar])
                    
                    if classifier:
                        sidebar_ids = [video.videoId for video in sidebar]
                        sidebar_csv = metadata_extractor.extract_metadata_batch(sidebar_ids, filename=f"metadata_sidebar_round_{round_num}.csv")
                        logger.info(f"Sidebar metadata saved to: {sidebar_csv}")
                        sidebar_df = classifier.classify_from_csv(sidebar_csv)
                        sidebar_scores = classifier.classify_video_list(sidebar, sidebar_df)
                        sidebar_harmful_count = sum(1 for score in sidebar_scores if score > harm_threshold)
                        sidebar_harmful_pct = (sidebar_harmful_count / len(sidebar)) * 100 if sidebar else 0
                        logger.info(f"Sidebar harmful content: {sidebar_harmful_pct:.2f}%")
                        
                        for video, score in zip(sidebar, sidebar_scores):
                            if score <= harm_threshold and video.videoId not in [v.videoId for v in harmless_reservoir]:
                                harmless_reservoir.append(video)
                        logger.info(f"Harmless reservoir size after sidebar update: {len(harmless_reservoir)}")
                    
                    if focus == "up-next":
                        current_recommendations = sidebar
                
                round_result = {
                    "round": round_num,
                    "timestamp": datetime.now().isoformat(),
                    "harmful_percentage": harmful_pct,
                    "sidebar_harmful_percentage": sidebar_harmful_pct if focus in ["up-next", "both"] else None,
                    "selected_video": selected_video.videoId
                }
                
            except Exception as e:
                logger.error(f"Error watching video: {str(e)}\n{traceback.format_exc()}")
                round_result = {
                    "round": round_num,
                    "timestamp": datetime.now().isoformat(),
                    "harmful_percentage": harmful_pct,
                    "error": str(e)
                }
            
            puppet["rounds"].append(round_result)
            puppet["harmful_exposure"].append({
                "round": round_num,
                "harmful_percentage": harmful_pct,
                "sidebar_harmful_percentage": sidebar_harmful_pct if focus in ["up-next", "both"] else None
            })
            
            time.sleep(2)
        
        add_action(puppet, "intervention_end")
        generate_visualization(puppet, os.path.join(experiment_dir, "visualizations"))
        save_results(puppet, os.path.join(output_dir, "results"))
        
        logger.info(f"Exiting run_intervention with puppet: {puppet}")
        return puppet
        
    except Exception as e:
        logger.error(f"Error running intervention: {str(e)}\n{traceback.format_exc()}")
        if puppet.get("standalone", False) and "driver" in puppet:
            puppet["driver"].close()
        puppet["error"] = str(e)
        save_results(puppet, os.path.join(output_dir, "results"))
        logger.info(f"Exiting run_intervention with puppet after error: {puppet}")
        return puppet

def add_action(puppet, action, params=None):
    puppet["actions"].append({
        "action": action,
        "params": params,
        "timestamp": datetime.now().isoformat()
    })

def save_recommendations_to_csv(puppet_id, round_num, source, recommendations, output_dir):
    csv_path = os.path.join(output_dir, f"{puppet_id}_round{round_num}_{source}.csv")
    data = []
    for video in recommendations:
        item = {'video_id': video.videoId, 'round': round_num, 'source': source}
        if hasattr(video, 'title'):
            item['title'] = video.title
        if hasattr(video, 'channel'):
            item['channel'] = video.channel
        if hasattr(video, 'description'):
            item['description'] = video.description
        data.append(item)
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

def save_results(puppet, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    results = {k: v for k, v in puppet.items() if k != 'driver'}
    results_path = os.path.join(output_dir, f"{puppet['puppetId']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

def generate_visualization(puppet, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    exposure_data = puppet.get('harmful_exposure', [])
    if not exposure_data:
        return
    rounds = [d.get('round', i+1) for i, d in enumerate(exposure_data)]
    harmful_percentages = [d.get('harmful_percentage', 0) for d in exposure_data]
    sidebar_harmful = [d.get('sidebar_harmful_percentage', 0) for d in exposure_data if d.get('sidebar_harmful_percentage') is not None]
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, harmful_percentages, 'b-o', linewidth=2, label='Main Recommendations')
    if sidebar_harmful:
        sidebar_rounds = rounds[:len(sidebar_harmful)]
        plt.plot(sidebar_rounds, sidebar_harmful, 'r-o', linewidth=2, label='Sidebar')
    plt.title(f"Harmful Content Exposure - {puppet['puppetId']}", fontsize=14)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Harmful Content (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{puppet['puppetId']}_exposure_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"), dpi=300, bbox_inches='tight')

if __name__ == "__main__":
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
    parser.add_argument("--harm-threshold", type=float, default=0.5, help="Harm classification threshold")
    parser.add_argument("--focus", default="both", choices=["homepage", "up-next", "both"],
                        help="Focus on homepage or up-next recommendations")
    parser.add_argument("--training", nargs="+", help="List of training video IDs")
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
        "standalone": True
    }
    run_intervention(args_dict)