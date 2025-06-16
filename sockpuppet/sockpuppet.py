import os
import json
import time
import logging
from datetime import datetime
from ytdriver import YTDriver, Video, VideoUnavailableException
import sys
import requests
from time import perf_counter

# Constants
OUTPUT_DIR = "/output"
LOG_DIR = "/logs"
MONITOR_URL = "http://gba.cs.ucdavis.edu:5000"

def init_puppet(puppetId, profile_dir):
    puppet = {
        "driver": YTDriver(profile_dir=profile_dir, use_virtual_display=True),
        "puppetId": puppetId,
        "actions": [],
        "start_time": datetime.now()
    }
    return puppet

def make_url(videoId):
    return "https://youtube.com/watch?v=" + videoId

def add_action(puppet, action, params=None):
    action = {
        "action": action,
        "params": params,
        "timestamp": datetime.now().isoformat()
    }
    logging.info(f"Action: {action}, Params: {params}, Timestamp: {action['timestamp']}")
    puppet["actions"].append(action)

def get_recommendations(puppet, focus, round_num=None):
    max_retries = 3
    retry_delay = 3
    for attempt in range(max_retries):
        try:
            if focus == "homepage":
                recommendations = puppet["driver"].get_homepage_recommendations(scroll_times=6)[:25]
            else:
                recommendations = puppet["driver"].get_upnext_recommendations(topn=12)
            video_ids = [vid.videoId for vid in recommendations]
            if not video_ids:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise ValueError(f"No recommendations returned for {focus} after {max_retries} attempts")
            add_action(puppet, f"get_{focus}_recommendations{'_' + str(round_num) if round_num else ''}", video_ids)
            return video_ids
        except Exception as e:
            logging.error(f"Attempt {attempt + 1}/{max_retries}: Error fetching recommendations: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            raise

def watch(puppet, video: Video, duration):
    try:
        puppet["driver"].play(video.url, duration=duration)
    except Exception as e:
        logging.error(f"Video {video.videoId} is unavailable: {e}")
        logging.info(f"Skipping unavailable video {video.videoId}l")
        add_action(puppet, "watch", {"videoId": video.videoId, "error": str(e)})
    else:
        add_action(puppet, "watch", video.videoId)

def save_puppet(puppet, args):
    js = {
        "puppet_id": puppet["puppetId"],
        "start_time": str(puppet["start_time"]),
        "end_time": str(datetime.now()),
        "actions": puppet["actions"],
        "args": args
    }
    puppet_file = os.path.join(args["outputDir"], "puppets", puppet["puppetId"] + ".json")
    os.makedirs(os.path.dirname(puppet_file), exist_ok=True)
    with open(puppet_file, "w") as f:
        json.dump(js, f, indent=4)
    logging.info(f"Saved puppet state to {puppet_file}")

def train(puppet, args):
    start = perf_counter()
    logging.info(f"Starting training for puppet {puppet['puppetId']}")
    add_action(puppet, "training_start")
    training = args.get("training", [])
    for videoId in training[:int(args["trainingN"])]:
        video = Video(None, make_url(videoId))
        start_watch = perf_counter()
        watch(puppet, video, args["duration"])
        end_watch = perf_counter()
        logging.info(f"Watched video {videoId} for {args['duration']} seconds in {end_watch - start_watch:.2f} seconds")
    add_action(puppet, "training_end")
    end = perf_counter()
    logging.info(f"Training completed for puppet {puppet['puppetId']} in {end - start:.2f} seconds")

def intervention(puppet, args):

    intervention_start = perf_counter()
    logging.info(f"Starting intervention for puppet {puppet['puppetId']}")
    add_action(puppet, "intervention_start")
    rounds = int(args.get("rounds", 10))
    focus = args.get("focus", "homepage")
    intervention_type = args.get("intervention_type", "downrank")
    # puppet_shared_dir = os.path.join(SHARED_DIR, puppet["puppetId"])
    # os.makedirs(puppet_shared_dir, exist_ok=True)

    # Intervention rounds
    for round_num in range(1, rounds + 1):
        add_action(puppet, f"round_{round_num}_start")
        
        # Fecth and dump the recommendations for this round
        start = perf_counter()
        for _ in range(10):
            try:
                recs = get_recommendations(puppet, focus, round_num)
                logging.info(f"Intervention(): Initializing round {round_num}")
                response = requests.post(f"{MONITOR_URL}/initialize_round", json={
                    "puppet_id": puppet["puppetId"],
                    "round_num": round_num,
                    "recommendations": recs
                })
            except Exception as e:
                logging.error(f"Could not contact monitor on {MONITOR_URL}: {e}")
                logging.info(f"Retrying in 5 seconds...")
                time.sleep(60)
                continue
            if response.status_code == 200:
                break
            elif response.status_code == 202:
                time.sleep(60)
            else:
                logging.error(f"Failed to initialize for round {round_num}: {response.text}")
                logging.info(f"Retrying in 60 seconds...")
                time.sleep(60)

        # with open(os.path.join(puppet_shared_dir, f"recommendations_{round_num}.txt"), "w") as f:
        #     f.write("\n".join(recs))
        logging.info(f"Getting recommendations took {perf_counter() - start:.2f} seconds")

        # Start preprocessing for this round
        start = perf_counter()
        for _ in range(10):
            try:
                logging.info(f"Intervention(): Starting preprocessing for round {round_num}")
                response = requests.post(f"{MONITOR_URL}/start_preprocess", json={
                    "puppet_id": puppet["puppetId"],
                    "round_num": round_num,
                    "intervention_type": intervention_type,
                    "focus": focus
                })
                logging.info(f"Intervention(): requests for preprocess")
            except Exception as e:
                logging.error(f"Could not contact monitor on {MONITOR_URL}: {e}")
                logging.info(f"Retrying in 5 seconds...")
                time.sleep(60)
                continue
            if response.status_code == 200:
                break
            elif response.status_code == 202:
                time.sleep(60)
            else:
                logging.error(f"Failed to start preprocessing for round {round_num}: {response.text}")
                logging.info(f"Retrying in 60 seconds...")
                time.sleep(60)
        logging.info(f"Start preprocess took {perf_counter() - start:.2f} seconds")
        
    
        # Wait for preprocessing to complete and get recommendations
        start = perf_counter()
        for _ in range(10):
            try:
                logging.info(f"Intervention():get recommendations for round {round_num}")
                response = requests.post(f"{MONITOR_URL}/get_recommendations", json={
                    "puppet_id": puppet["puppetId"],
                    "round_num": round_num,
                    "intervention_type": intervention_type,
                })
                logging.info(f"Intervention(): request recommendations for round {round_num}")
            except Exception as e:
                logging.error(f"Could not contact monitor on {MONITOR_URL}: {e}")
                time.sleep(60)
                logging.info(f"Retrying in 5 seconds...")
                continue
            if response.status_code == 200:
                data = response.json()
                next_video = data.get("next_video")
                if not next_video:
                    logging.error(f"No next video received for round {round_num}")
                    break
                logging.info(f"Received next video {next_video} for round {round_num}")
                break
            elif response.status_code == 202:
                time.sleep(60)
            else:
                logging.error(f"Failed to get recommendation in preprocess for round {round_num}: {response.text}")
                logging.info(f"Retrying in 60 seconds...")
                time.sleep(60)
            # logging.info(f"Waiting for preprocessing to complete for round {round_num}")
            # time.sleep(60)
            
        logging.info(f"Get recommendations took {perf_counter() - start:.2f} seconds")

        # Watch the video
        start = perf_counter()
        logging.info(f"Intervention():Watching video {next_video} for round {round_num}")
        selected_video = Video(None, make_url(next_video))
        watch(puppet, selected_video, args["duration"])
        logging.info(f"Intervention():Finished watching video {next_video} for round {round_num}")
        logging.info(f"Watching video took {perf_counter() - start:.2f} seconds")
        
        # Signal round completion
        start = perf_counter()
        logging.info(f"Intervention():Signaling completion for round {round_num}")
        for _ in range(10):
            try:
                response = requests.post(f"{MONITOR_URL}/complete_round", json={
                    "puppet_id": puppet["puppetId"],
                    "round_num": round_num
                })
                logging.info(f"Intervention(): requests for complete round")
            except Exception as e:
                logging.error(f"Could not contact monitor on {MONITOR_URL}: {e}")
                time.sleep(5)
                logging.info(f"Retrying in 5 seconds...")
                continue
            if response.status_code == 200:
                break
            elif response.status_code == 500:
                logging.error(f"Round {round_num} completion not acknowledged: {response.text}")
                return
            elif response.status_code == 202:
                time.sleep(60)
            else:
                logging.error(f"Failed to signal preprocessing completion for round {round_num}: {response.text}")
                logging.info(f"Retrying in 60 seconds...")
                time.sleep(60)
            logging.info(f"Signaled completion for round {round_num}")

        logging.info(f"Round {round_num} cleanup completed in {perf_counter() - start:.2f} seconds")
        add_action(puppet, f"round_{round_num}_end")

    add_action(puppet, "intervention_end")
    end = perf_counter()
    logging.info(f"Intervention completed for puppet {puppet['puppetId']} in {end - intervention_start:.2f} seconds")

if __name__ == "__main__":
    args = json.loads(sys.argv[1])
    args["outputDir"] = args.get("outputDir", "/output")
    log_filename = f"{args['puppetId']}_{args.get('steps', 'unknown')}.log"
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=os.path.join(LOG_DIR, log_filename),
        level=logging.INFO,
        filemode='w'
    )

    profile_dir = os.path.join(args["outputDir"], "profiles", args["puppetId"])
    os.makedirs(os.path.dirname(profile_dir), exist_ok=True)
    
    # Load pre-trained puppet state for intervention-only or initialize for combined/train
    puppet_state_file = os.path.join(args["outputDir"], "puppets", f"{args['puppetId']}.json")
    if args["steps"] == "intervention" and os.path.exists(puppet_state_file):
        with open(puppet_state_file, "r") as f:
            puppet_state = json.load(f)
        puppet = init_puppet(args["puppetId"], profile_dir)
        puppet["actions"] = puppet_state["actions"]
        puppet["start_time"] = datetime.strptime(puppet_state["start_time"], "%Y-%m-%d %H:%M:%S.%f")
        logging.info(f"Loaded pre-trained puppet state for {args['puppetId']}")
    else:
        puppet = init_puppet(args["puppetId"], profile_dir)

    steps = args["steps"]

    if steps == "train":
        train(puppet, args)
    elif steps == "intervention":
        intervention(puppet, args)
    elif steps == "combined":
        train(puppet, args)
        intervention(puppet, args)

    save_puppet(puppet, args)
    puppet["driver"].close()
    logging.info("Sock puppet finished")