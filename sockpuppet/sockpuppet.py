import os
import json
import time
import logging
from datetime import datetime
from ytdriver import YTDriver, Video
import sys
import requests
from pathlib import Path
from time import perf_counter

session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=200, pool_maxsize=200) 
session.mount('http://', adapter) 
session.mount('https://', adapter)


# Constants - updated to work with stage-specific directories
MONITOR_URL = "http://host.docker.internal:5005"

# Directory name constants (must match orchestrator)
OUTPUT_DIR_NAME = "debug-output"
LOGS_DIR_NAME = "logs"
SHARED_DIR_NAME = "debug-shared"
PROFILES_DIR_NAME = "profiles"
PUPPETS_DIR_NAME = "puppets"

def get_stage_dirs(output_dir: str):
    """Extract stage and determine directory structure"""
    # output_dir will be like "/train-debug-output" or "/intervention-debug-output"
    if f"train-{OUTPUT_DIR_NAME}" in output_dir:
        stage = "train"
        shared_dir = f"/train-{SHARED_DIR_NAME}"
        log_dir = f"/train-{LOGS_DIR_NAME}"
    elif f"intervention-{OUTPUT_DIR_NAME}" in output_dir:
        stage = "intervention" 
        shared_dir = f"/intervention-{SHARED_DIR_NAME}"
        log_dir = f"/intervention-{LOGS_DIR_NAME}"
    else:
        # Fallback for backward compatibility
        stage = "unknown"
        shared_dir = f"/{SHARED_DIR_NAME}"
        log_dir = f"/{LOGS_DIR_NAME}"
    
    return {
        "stage": stage,
        "output_dir": output_dir,
        "shared_dir": shared_dir,
        "log_dir": log_dir
    }

def init_puppet(puppetId, profile_dir):
    """Initialize puppet with driver and tracking data"""
    puppet = {
        "driver": YTDriver(profile_dir=profile_dir, use_virtual_display=True),
        "puppetId": puppetId,
        "actions": [],
        "start_time": datetime.now()
    }
    return puppet

def make_url(videoId):
    """Create YouTube URL from video ID"""
    return "https://youtube.com/watch?v=" + videoId

def add_action(puppet, action, params=None):
    """Add action to puppet's action history"""
    logging.info(f"Action: {action}, Params: {params}")
    puppet["actions"].append({
        "action": action,
        "params": params,
        "timestamp": datetime.now().isoformat()
    })

def get_recommendations(puppet, focus, round_num=None, dirs=None):
    """Get recommendations from YouTube with retry logic"""
    max_retries = 3
    retry_delay = 3
    
    for attempt in range(max_retries):
        try:
            if focus == "homepage":
                recommendations = puppet["driver"].get_homepage_recommendations(scroll_times=6)[:25]
            else:  # upnext
                recommendations = puppet["driver"].get_upnext_recommendations(topn=12)
            
            video_ids = [vid.videoId for vid in recommendations]
            
            if not video_ids:
                logging.warning(f"Attempt {attempt + 1}/{max_retries}: No recommendations for {focus}")
                # screenshot_dir = os.path.join(dirs["output_dir"], "screenshots", puppet["puppetId"])
                save_dir = Path(dirs["shared_dir"]) / puppet["puppetId"]

                os.makedirs(save_dir, exist_ok=True)
                
                screenshot_name = f"round_{round_num}_focus_{focus}_attempt_{attempt + 1}.png"
                screenshot_path = os.path.join(save_dir, screenshot_name)
                
                logging.info(f"Saving screenshot for failed {focus} recommendations to {screenshot_path}")
                
                try:
                    # Save screenshot
                    screenshot_success = puppet["driver"].save_screenshot(screenshot_path)
                    if screenshot_success:
                        logging.info(f"Screenshot saved successfully to {screenshot_path}")
                    else:
                        logging.error(f"Failed to save screenshot to {screenshot_path}")
                        
                except Exception as e:
                    logging.error(f"Error saving debug information: {e}")
                    
                if attempt < max_retries - 1:
                    logging.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                    
                raise ValueError(f"No recommendations returned for {focus} after {max_retries} attempts")
            
            action_name = f"get_{focus}_recommendations"
            if round_num is not None:
                action_name += f"_{round_num}"
            
            add_action(puppet, action_name, video_ids)
            return video_ids
            
        except Exception as e:
            logging.error(f"Attempt {attempt + 1}/{max_retries}: Error fetching recommendations: {str(e)}")
            
            save_dir = Path(dirs["shared_dir"]) / puppet["puppetId"]
            os.makedirs(save_dir, exist_ok=True)
            
            screenshot_name = f"round_{round_num}_focus_{focus}_attempt_{attempt + 1}_error.png"
            screenshot_path = os.path.join(save_dir, screenshot_name)
            
            try:
                screenshot_success = puppet["driver"].save_screenshot(screenshot_path)
                if screenshot_success:
                    logging.error(f"Error screenshot saved to {screenshot_path}")
                else:
                    logging.error(f"Failed to save error screenshot")
            except Exception as screenshot_error:
                logging.error(f"Error saving screenshot: {screenshot_error}")
            
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            raise

def watch(puppet, video: Video, duration):
    """Watch a video with error handling"""
    try:
        puppet["driver"].play(video, duration=duration)
        add_action(puppet, "watch", video.videoId)
        logging.info(f"Successfully watched video {video.videoId}")
    except Exception as e:
        logging.error(f"Video {video.videoId} is unavailable: {e}")
        add_action(puppet, "watch", {"videoId": video.videoId, "error": str(e)})

def save_puppet(puppet, args, dirs, initial_recommendations=None):
    """Save puppet state to JSON file"""
    puppet_data = {
        "puppet_id": puppet["puppetId"],
        "start_time": str(puppet["start_time"]),
        "end_time": str(datetime.now()),
        "actions": puppet["actions"],
        "args": args,
        "stage": dirs["stage"]
    }
    
    # Add initial recommendations if provided (from training stage)
    if initial_recommendations:
        puppet_data["initial_recommendations"] = initial_recommendations
        logging.info(f"Saved initial recommendations: {len(initial_recommendations)} sets")
    
    puppet_file = Path(args["outputDir"]) / PUPPETS_DIR_NAME / f"{puppet['puppetId']}.json"
    puppet_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(puppet_file, "w") as f:
        json.dump(puppet_data, f, indent=4)
    
    logging.info(f"Saved puppet state to {puppet_file}")

def train(puppet, args):
    """Training phase - watch training videos and capture initial recommendations"""
    start = perf_counter()  
    logging.info(f"Starting training for puppet {puppet['puppetId']}")
    add_action(puppet, "training_start")
    
    training_videos = args.get("training", [])
    total_videos = min(len(training_videos), int(args["trainingN"]))
    
    logging.info(f"Training with {total_videos} videos ({args.get('harmful_percentage', 0)}% harmful)")
    
    for i, videoId in enumerate(training_videos[:total_videos]):
        logging.info(f"Training video {i+1}/{total_videos}: {videoId}")
        video = Video(None, make_url(videoId))
        start_watch = perf_counter()
        watch(puppet, video, args["duration"])
        end_watch = perf_counter()
        logging.info(f"Watched video {videoId} for {args['duration']} seconds in {end_watch - start_watch:.2f} seconds")
       
    # After training, capture initial recommendations for both homepage and upnext
    logging.info("Capturing initial recommendations after training...")
    initial_recommendations = {}

    start = perf_counter()
    try:
        # Get upnext recommendations (should work since we just watched a video)
        upnext_recs = puppet["driver"].get_upnext_recommendations(topn=12)
        upnext_video_ids = [vid.videoId for vid in upnext_recs]
        initial_recommendations["upnext"] = upnext_video_ids
        add_action(puppet, "capture_initial_upnext_recommendations", upnext_video_ids)
        logging.info(f"Captured {len(upnext_video_ids)} initial upnext recommendations")
    except Exception as e:
        logging.error(f"Failed to capture upnext recommendations: {e}")
        initial_recommendations["upnext"] = []
    end = perf_counter()
    logging.info(f"Captured initial upnext recommendations in {end - start:.2f} seconds")
    
    start = perf_counter()
    try:
        # Get homepage recommendations
        homepage_recs = puppet["driver"].get_homepage_recommendations(scroll_times=6)[:25]
        homepage_video_ids = [vid.videoId for vid in homepage_recs]
        initial_recommendations["homepage"] = homepage_video_ids
        add_action(puppet, "capture_initial_homepage_recommendations", homepage_video_ids)
        logging.info(f"Captured {len(homepage_video_ids)} initial homepage recommendations")
    except Exception as e:
        logging.error(f"Failed to capture homepage recommendations: {e}")
        initial_recommendations["homepage"] = []
    end = perf_counter()
    logging.info(f"Captured initial homepage recommendations in {end - start:.2f} seconds")
    
    add_action(puppet, "training_end")
    end = perf_counter()
    logging.info(f"Training completed for puppet {puppet['puppetId']} in {end - start:.2f} seconds")
    
    # Return initial recommendations to be saved
    return initial_recommendations

def intervention(puppet, args, dirs, initial_recommendations=None):
    """Intervention phase - test recommendation interventions"""
    intervention_start = perf_counter()
    logging.info(f"Starting intervention for puppet {puppet['puppetId']}")
    add_action(puppet, "intervention_start")
    
    rounds = int(args.get("rounds", 10))
    focus = args.get("focus", "homepage")
    intervention_type = args.get("intervention_type", "downrank")
    
    puppet_shared_dir = Path(dirs["shared_dir"]) / puppet["puppetId"]
    puppet_shared_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Intervention config: {rounds} rounds, focus={focus}, type={intervention_type}")
    logging.info(f"Shared directory: {puppet_shared_dir}")

    # Run intervention rounds
    for round_num in range(1, rounds + 1):
        logging.info(f"=== Round {round_num}/{rounds} ===")
        add_action(puppet, f"round_{round_num}_start")
        start = perf_counter()
        try:
            # For the first round, use stored initial recommendations if available
            if round_num == 1 and initial_recommendations and focus in initial_recommendations:
                recommendations = initial_recommendations[focus]
                logging.info(f"Using stored initial {focus} recommendations for round 1: {len(recommendations)} videos")
                add_action(puppet, f"use_initial_{focus}_recommendations_1", recommendations)
            else:
                # Get recommendations for this round - PASS dirs parameter here
                logging.info(f"Fetching {focus} recommendations for round {round_num}")
                recommendations = get_recommendations(puppet, focus, round_num, dirs)
            
            # Save recommendations to shared directory
            rec_file = puppet_shared_dir / f"recommendations_{round_num}.txt"
            with open(rec_file, "w") as f:
                f.write("\n".join(recommendations))
            
            logging.info(f"Saved {len(recommendations)} recommendations to {rec_file} in {perf_counter() - start:.2f} seconds")

            start = perf_counter()
            # Start preprocessing via monitor
            try:
                logging.info(f"Starting preprocessing for round {round_num}")
                response = session.post(f"{MONITOR_URL}/start_preprocess", json={
                    "puppet_id": puppet["puppetId"],
                    "round_num": round_num,
                    "intervention_type": intervention_type,
                    "focus": focus
                })
                
            except Exception as e:
                logging.error(f"Could not contact monitor on {MONITOR_URL}: {e}")
                return
            if response.status_code != 200:
                logging.error(f"Failed to start preprocess for round {round_num}: {response.text}")
                break
            logging.info(f"Start preprocess took {perf_counter() - start:.2f} seconds")
            
            # Wait for preprocessing to complete and get recommendations
            start = perf_counter()
            while True:
                try:
                    response = session.post(f"{MONITOR_URL}/get_recommendations", json={
                        "puppet_id": puppet["puppetId"],
                        "round_num": round_num,
                        "intervention_type": intervention_type,
                    })
                        
                except Exception as e:
                    logging.error(f"Could not contact monitor on {MONITOR_URL}: {e}")
                    return
                if response.status_code == 200:
                    data = response.json()
                    next_video = data.get("next_video")
                    if not next_video:
                        logging.error(f"No next video received for round {round_num}")
                        break
                    logging.info(f"Received next video {next_video} for round {round_num}")
                    break
                elif response.status_code == 500:
                    logging.error(f"Preprocessing failed for round {round_num}: {response.text}")
                    return 
                logging.info(f"Waiting for preprocessing to complete for round {round_num}")
                time.sleep(2)
                
            logging.info(f"Get recommendations took {perf_counter() - start:.2f} seconds")


            # Watch the selected video
            start = perf_counter()
            if next_video:
                logging.info(f"Watching video {next_video} for round {round_num}")
                selected_video = Video(None, make_url(next_video))
                watch(puppet, selected_video, args["duration"])
                logging.info(f"Watching video took {perf_counter() - start:.2f} seconds")
        
                # Signal round completion to monitor
                start = perf_counter()
                try:
                    response = session.post(f"{MONITOR_URL}/complete_round", json={
                        "puppet_id": puppet["puppetId"],
                        "round_num": round_num
                    })
                    
                    if response.status_code == 200:
                        logging.info(f"Round {round_num} completion acknowledged")
                    else:
                        logging.warning(f"Round {round_num} completion not acknowledged: {response.text}")
                        
                except requests.exceptions.RequestException as e:
                    logging.warning(f"Could not signal completion for round {round_num}: {e}")
            else:
                logging.error(f"No video to watch for round {round_num}")
            
        except Exception as e:
            logging.error(f"Error in round {round_num}: {e}")
        logging.info(f"Round {round_num} cleanup completed in {perf_counter() - start:.2f} seconds")
        add_action(puppet, f"round_{round_num}_end")
        
        
    end = perf_counter()
    add_action(puppet, "intervention_end")
    logging.info(f"Intervention completed for puppet {puppet['puppetId']} in {end - intervention_start:.2f} seconds")

def load_existing_puppet_state(puppet_state_file: Path, puppet_id: str):
    """Load existing puppet state for intervention stage"""
    try:
        with open(puppet_state_file, "r") as f:
            puppet_state = json.load(f)
        
        # Validate that this is the right puppet
        if puppet_state.get("puppet_id") != puppet_id.split("_homepage_")[0].split("_upnext_")[0]:
            # Extract base puppet ID by removing intervention suffixes
            base_id = puppet_id
            for suffix in ["_homepage_none", "_homepage_downrank", "_homepage_replace",
                          "_upnext_none", "_upnext_downrank", "_upnext_replace"]:
                if base_id.endswith(suffix):
                    base_id = base_id[:-len(suffix)]
                    break
            
            if puppet_state.get("puppet_id") != base_id:
                logging.warning(f"Puppet ID mismatch: expected {base_id}, got {puppet_state.get('puppet_id')}")
        
        return puppet_state
    except Exception as e:
        logging.error(f"Error loading puppet state from {puppet_state_file}: {e}")
        return None

def setup_logging(puppet_id: str, stage: str, log_dir: str):
    """Setup logging for the puppet"""
    log_filename = f"{puppet_id}_{stage}.log"
    log_path = Path(log_dir) / log_filename
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=str(log_path),
        level=logging.INFO,
        filemode='a'
    )
    
    # Also log to console for debugging
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(console_handler)
    
    logging.info(f"Logging initialized for puppet {puppet_id} in stage {stage}")

def main():
    """Main entry point for sockpuppet"""
    if len(sys.argv) != 2:
        logging.error("Usage: python sockpuppet.py '<json_args>'")
        sys.exit(1)
    
    try:
        args = json.loads(sys.argv[1])
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON arguments: {e}")
        sys.exit(1)
    
    # Get directory structure
    dirs = get_stage_dirs(args.get("outputDir", "/debug-output"))
    
    # Setup logging
    setup_logging(args["puppetId"], dirs["stage"], dirs["log_dir"])
    
    logging.info(f"Starting sockpuppet with stage: {dirs['stage']}")
    logging.info(f"Puppet ID: {args['puppetId']}")
    logging.info(f"Steps: {args.get('steps', 'unknown')}")
    logging.info(f"Args: {json.dumps(args, indent=2)}")
    
    # Setup profile directory
    profile_dir = Path(args["outputDir"]) / PROFILES_DIR_NAME / args["puppetId"]
    profile_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize or load puppet
    puppet_state_file = Path(args["outputDir"]) / PUPPETS_DIR_NAME / f"{args['puppetId']}.json"
    initial_recommendations = None
    
    if args["steps"] == "intervention" and puppet_state_file.exists():
        # Load existing puppet state for intervention
        logging.info(f"Loading existing puppet state from {puppet_state_file}")
        puppet_state = load_existing_puppet_state(puppet_state_file, args["puppetId"])
        
        puppet = init_puppet(args["puppetId"], str(profile_dir))
        
        if puppet_state:
            puppet["actions"] = puppet_state.get("actions", [])
            initial_recommendations = puppet_state.get("initial_recommendations", {})
            
            if initial_recommendations:
                logging.info(f"Loaded initial recommendations: homepage={len(initial_recommendations.get('homepage', []))}, upnext={len(initial_recommendations.get('upnext', []))}")
            
            try:
                puppet["start_time"] = datetime.strptime(
                    puppet_state["start_time"], 
                    "%Y-%m-%d %H:%M:%S.%f"
                )
            except (ValueError, KeyError):
                # Fallback for different datetime formats
                puppet["start_time"] = datetime.now()
            
            logging.info(f"Loaded {len(puppet['actions'])} previous actions")
        else:
            logging.warning("Could not load puppet state, starting fresh")
    else:
        # Initialize new puppet
        puppet = init_puppet(args["puppetId"], str(profile_dir))
        logging.info(f"Initialized new puppet with profile directory: {profile_dir}")
    
    # Execute the requested steps
    steps = args.get("steps", "combined")
    
    try:
        if steps == "train":
            initial_recommendations = train(puppet, args)
            # Save puppet state with initial recommendations
            save_puppet(puppet, args, dirs, initial_recommendations)
        elif steps == "intervention":
            intervention(puppet, args, dirs, initial_recommendations)
            # Save puppet state (no initial recommendations to add)
            save_puppet(puppet, args, dirs)
        elif steps == "combined":
            initial_recommendations = train(puppet, args)
            intervention(puppet, args, dirs, initial_recommendations)
            # Save puppet state with initial recommendations
            save_puppet(puppet, args, dirs, initial_recommendations)
        else:
            logging.error(f"Unknown steps: {steps}")
            sys.exit(1)
            
    except Exception as e:
        logging.error(f"Error during puppet execution: {e}", exc_info=True)
        # Save puppet state even on error
        try:
            save_puppet(puppet, args, dirs, initial_recommendations)
        except:
            pass
        sys.exit(1)
    
    finally:
        # Clean up
        try:
            puppet["driver"].close()
            logging.info("Driver closed successfully")
        except Exception as e:
            logging.warning(f"Error closing driver: {e}")
    
    logging.info("Sockpuppet execution completed successfully")

if __name__ == "__main__":
    main()