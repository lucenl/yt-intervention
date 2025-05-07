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
    
    logger.info(f"Puppet at start of run_intervention: {puppet}")
    
    # Extract parameters
    puppet_id = args["puppetId"]
    output_dir = args.get("outputDir", "/output")
    model_path = args.get("model_path", "/app/models/roberta_checkpoint")
    intervention_type = args.get("intervention_type", "none")
    selection_type = args.get("selection_type", "decay_weighted_random") 
    num_rounds = int(args.get("rounds", 10))
    watch_duration = int(args.get("duration", 30))
    harm_threshold = float(args.get("harm_threshold", 0.8))
    
    # Setup directories
    experiment_dir = os.path.join(output_dir, puppet_id)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Initialize metadata extractor
    metadata_extractor = MetadataExtractor(
        output_dir=experiment_dir,
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
        # Create harmless reservoir for replacement strategy
        harmless_reservoir = [] if intervention_type == 'replace' else None
        
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
            
            # Get metadata for classification
            metadata_csv = metadata_extractor.extract_metadata_batch(video_ids)
            
            # Classify videos if classifier is available
            harmful_pct = 0
            harm_scores = [0.0] * len(recommendations)
            if classifier:
                # Classify using RoBERTa model
                metadata_df = classifier.classify_from_csv(metadata_csv)
                
                # Get harm scores as a list in the same order as recommendations
                harm_scores = classifier.classify_video_list(recommendations, metadata_df)
                if len(harm_scores) != len(recommendations):
                    logger.error(f"Mismatch in number of scores ({len(harm_scores)}) and recommendations ({len(recommendations)})")
                    raise ValueError("Mismatch in classification scores and recommendations")
                
                # Log harm scores
                logger.info(f"Harm scores: {dict(zip(video_ids, harm_scores))}")
                
                # Calculate harmful percentage
                harmful_count = sum(1 for score in harm_scores if score > harm_threshold)
                harmful_pct = (harmful_count / len(recommendations)) * 100 if recommendations else 0
                source = "Homepage" if focus == "homepage" else "Up-Next" if focus == "up-next" else "Homepage"
                logger.info(f"{source} harmful content: {harmful_pct:.2f}%")
                
                # Apply intervention
                if intervention_type == 'downrank':
                    # Downrank harmful videos based on harm scores
                    video_scores = list(zip(recommendations, harm_scores))
                    sorted_pairs = sorted(video_scores, key=lambda x: x[1])  # Sort by harm score (ascending)
                    modified_recommendations = [video for video, _ in sorted_pairs]
                    
                    # Log reranking details
                    logger.info(f"Reranked recommendations: {[vid.videoId for vid in modified_recommendations]}")
                    
                elif intervention_type == 'replace' and harmless_reservoir:
                    # Replace harmful videos with harmless ones
                    modified_recommendations = recommendations.copy()
                    used_replacements = set()
                    
                    for i, (video, score) in enumerate(zip(recommendations, harm_scores)):
                        if score > harm_threshold:  # Harmful video
                            for replacement in harmless_reservoir:
                                if (replacement.videoId not in used_replacements and
                                    replacement.videoId not in [v.videoId for v in recommendations]):
                                    modified_recommendations[i] = replacement
                                    used_replacements.add(replacement.videoId)
                                    logger.info(f"Replaced harmful video {video.videoId} (score: {score}) with {replacement.videoId}")
                                    break
                else:
                    # No intervention
                    modified_recommendations = recommendations
            else:
                # No classifier, no intervention
                modified_recommendations = recommendations
            
            # Select video to watch
            if not modified_recommendations:
                logger.warning("No videos available to select. Skipping round.")
                puppet["harmful_exposure"].append({
                    "round": round_num,
                    "harmful_percentage": 0.0
                })
                continue
                
            if selection_type == 'decay_weighted_random':
                # Position-based weighting with harm score adjustment
                decay_factor = 0.9
                weights = [(1.0 - score) * (decay_factor ** i) for i, score in enumerate(harm_scores)]
                weights_sum = sum(weights)
                if weights_sum > 0:
                    normalized_weights = [w / weights_sum for w in weights]
                else:
                    normalized_weights = [1.0 / len(modified_recommendations)] * len(modified_recommendations)
                selected_index = np.random.choice(len(modified_recommendations), p=normalized_weights)
                selected_video = modified_recommendations[selected_index]
                
                # Log selection details
                logger.info(f"Decay-adjusted weights: {normalized_weights}")
                logger.info(f"Selected video index: {selected_index} with weight: {normalized_weights[selected_index]}")
                logger.info(f"Selected video: {selected_video.videoId}")
                
            elif selection_type == 'random':
                # Random selection
                selected_video = random.choice(modified_recommendations)
                logger.info(f"Selected video (random): {selected_video.videoId}")
                
            else:  # top or default
                # Pick top video
                selected_video = modified_recommendations[0]
                logger.info(f"Selected video (top): {selected_video.videoId}")
            
            # Add selection action
            add_action(puppet, "select_video", selected_video.videoId)
            
            # Watch the selected video
            try:
                logger.info(f"Watching video: {selected_video.videoId}")
                puppet["driver"].play(selected_video, duration=watch_duration)
                logger.info(f"Finished watching video: {selected_video.videoId}")
                add_action(puppet, "watch", selected_video.videoId)
                
                # Get new recommendations based on focus (for next round, handled in loop)
                sidebar_harmful_pct = 0
                if focus in ["up-next", "both"]:
                    sidebar = puppet["driver"].get_upnext_recommendations(topn=12)
                    logger.info(f"Got {len(sidebar)} sidebar recommendations")
                    add_action(puppet, "get_upnext_recommendations", [vid.videoId for vid in sidebar])
                    
                    # Get sidebar metadata and classify
                    if classifier:
                        sidebar_ids = [video.videoId for video in sidebar]
                        sidebar_csv = metadata_extractor.extract_metadata_batch(sidebar_ids)
                        sidebar_df = classifier.classify_from_csv(sidebar_csv)
                        sidebar_scores = classifier.classify_video_list(sidebar, sidebar_df)
                        sidebar_harmful_count = sum(1 for score in sidebar_scores if score > harm_threshold)
                        sidebar_harmful_pct = (sidebar_harmful_count / len(sidebar)) * 100 if sidebar else 0
                        logger.info(f"Sidebar harmful content: {sidebar_harmful_pct:.2f}%")
                        
                        # Update harmless reservoir if needed
                        if harmless_reservoir is not None:
                            for video, score in zip(sidebar, sidebar_scores):
                                if score <= harm_threshold:
                                    if video.videoId not in [v.videoId for v in harmless_reservoir]:
                                        harmless_reservoir.append(video)
                    
                    if focus == "up-next":
                        current_recommendations = sidebar
                
                # Store round results
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
            
            # Store round results
            puppet["rounds"].append(round_result)
            
            # Track harmful exposure
            puppet["harmful_exposure"].append({
                "round": round_num,
                "harmful_percentage": harmful_pct,
                "sidebar_harmful_percentage": sidebar_harmful_pct if focus in ["up-next", "both"] else None
            })
            
            # Short delay between rounds
            time.sleep(2)
        
        # Add end action
        add_action(puppet, "intervention_end")
        
        # Generate visualization of harmful exposure
        generate_visualization(puppet, os.path.join(experiment_dir, "visualizations"))
        
        # Save results
        save_results(puppet, os.path.join(output_dir, "results"))
        
        return puppet
        
    except Exception as e:
        logger.error(f"Error running intervention: {str(e)}\n{traceback.format_exc()}")
        
        # Clean up if standalone mode
        if puppet.get("standalone", False) and "driver" in puppet:
            puppet["driver"].close()
            
        # Add error and save partial results
        puppet["error"] = str(e)
        save_results(puppet, os.path.join(output_dir, "results"))
        
        return puppet

def add_action(puppet, action, params=None):
    """
    Add an action to the puppet's action log
    
    Args:
        puppet: Puppet dictionary
        action: Name of the action
        params: Optional parameters for the action
    """
    puppet["actions"].append({
        "action": action,
        "params": params,
        "timestamp": datetime.now().isoformat()
    })

def save_recommendations_to_csv(puppet_id, round_num, source, recommendations, output_dir):
    """
    Save recommendations to a CSV file
    
    Args:
        puppet_id: ID of the puppet
        round_num: Current round number
        source: Source of recommendations ('homepage' or 'sidebar')
        recommendations: List of Video objects
        output_dir: Directory to save the CSV
    """
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
        
    # Save as CSV
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

def save_results(puppet, output_dir):
    """
    Save experiment results to a JSON file
    
    Args:
        puppet: Puppet dictionary with experiment results
        output_dir: Directory to save the results
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {k: v for k, v in puppet.items() if k != 'driver'}
    results_path = os.path.join(
        output_dir, 
        f"{puppet['puppetId']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

def generate_visualization(puppet, output_dir):
    """
    Generate a visualization of harmful content exposure over time
    
    Args:
        puppet: Puppet dictionary with experiment results
        output_dir: Directory to save the visualization
    """
    os.makedirs(output_dir, exist_ok=True)
    exposure_data = puppet.get('harmful_exposure', [])
    if not exposure_data:
        return
    rounds = [d.get('round', i+1) for i, d in enumerate(exposure_data)]
    harmful_percentages = [d.get('harmful_percentage', 0) for d in exposure_data]
    sidebar_harmful = [d.get('sidebar_harmful_percentage', 0) for d in exposure_data 
                       if d.get('sidebar_harmful_percentage') is not None]
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
    plt.savefig(os.path.join(output_dir, f"{puppet['puppetId']}_exposure.png"), dpi=300, bbox_inches='tight')

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
        "standalone": True
    }
    run_intervention(args_dict)