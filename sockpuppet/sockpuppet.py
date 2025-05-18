import os
import json
import time
import logging
from datetime import datetime
from ytdriver import YTDriver, Video, VideoUnavailableException
import sys
import requests

# Constants
SHARED_DIR = "/shared"
OUTPUT_DIR = "/output"
LOG_DIR = "/logs"
MONITOR_URL = "http://host.docker.internal:5000"

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
    logging.info(f"Action: {action}, Params: {params}")
    puppet["actions"].append({
        "action": action,
        "params": params,
        "timestamp": datetime.now().isoformat()
    })

def get_recommendations(puppet, focus, round_num=None):
    max_retries = 3
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            if focus == "homepage":
                recommendations = puppet["driver"].get_homepage_recommendations(scroll_times=6)[:25]
            else:
                recommendations = puppet["driver"].get_upnext_recommendations(topn=12)
            video_ids = [vid.videoId for vid in recommendations]
            if not video_ids:
                logging.warning(f"Attempt {attempt + 1}/{max_retries}: No recommendations for {focus}")
                screenshot_dir = os.path.join(OUTPUT_DIR, "screenshots", puppet["puppetId"])
                os.makedirs(screenshot_dir, exist_ok=True)
                logging.info(f"Saving screenshot for unavailable video {video_ids}")
                screenshot_name = f"{error}_{round_num}_recs_{attempt + 1}.png"
                screenshot_path = os.path.join(screenshot_dir, screenshot_name)
                logging.info(f"Saving screenshot to {screenshot_path}")
                screenshot = puppet["driver"].save_screenshot(screenshot_path)
                logging.info(f"Screenshot saved to {screenshot_path}")
                if screenshot:
                    logging.error(f"Saved screenshot to {screenshot_path}")
                else:
                    logging.error(f"Failed to save screenshot for unavailable video {video.videoId}")
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
        puppet["driver"].play(video, duration=duration)
    except Exception as e:
        logging.error(f"Video {video.videoId} is unavailable: {e}")
        screenshot_dir = os.path.join(OUTPUT_DIR, "screenshots", puppet["puppetId"])
        logging.info(f"Saving screenshot for unavailable video {video.videoId}")
        os.makedirs(screenshot_dir, exist_ok=True)
        logging.info(f"Saving screenshot to {screenshot_dir}")
        screenshot_name = f"error_{video.videoId}.png"
        screenshot_path = os.path.join(screenshot_dir, screenshot_name)
        logging.info(f"Saving screenshot to {screenshot_path}")
        puppet["driver"].save_screenshot(screenshot_path)
        logging.info(f"Screenshot saved to {screenshot_path}")
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
    logging.info(f"Starting training for puppet {puppet['puppetId']}")
    add_action(puppet, "training_start")
    training = args.get("training", [])
    for videoId in training[:int(args["trainingN"])]:
        video = Video(None, make_url(videoId))
        watch(puppet, video, args["duration"])
    add_action(puppet, "training_end")

def intervention(puppet, args):
    logging.info(f"Starting intervention for puppet {puppet['puppetId']}")
    add_action(puppet, "intervention_start")
    rounds = int(args.get("rounds", 10))
    focus = args.get("focus", "homepage")
    intervention_type = args.get("intervention_type", "downrank")
    puppet_shared_dir = os.path.join(SHARED_DIR, puppet["puppetId"])
    os.makedirs(puppet_shared_dir, exist_ok=True)

    # Bootstrap: Fetch initial recommendations
    video_ids = get_recommendations(puppet, focus)
    with open(os.path.join(puppet_shared_dir, "recommendations_0.txt"), "w") as f:
        f.write("\n".join(video_ids))

    # Start preprocessing for round 0
    try:
        response = requests.post(f"{MONITOR_URL}/start_preprocess", json={
            "puppet_id": puppet["puppetId"],
            "round_num": 0,
            "intervention_type": intervention_type,
            "focus": focus
        }, timeout=30)
    except Exception as e:
        logging.error(f"Could not contact monitor on {MONITOR_URL}: {e}")
        return
    if response.status_code != 200:
        logging.error(f"Failed to start preprocess for round 0: {response.text}")
        return
    logging.info("Started preprocessing for round 0")

    # Wait for preprocessing to complete
    while True:
        try:
            response = requests.post(f"{MONITOR_URL}/get_recommendations", json={
                "puppet_id": puppet["puppetId"],
                "round_num": 0,
                "intervention_type": intervention_type,
            }, timeout=30)
            # logging.info(f"Polling /get_recommendations, status: {response.status_code}, text: {response.text}")
        except Exception as e:
            logging.error(f"Could not contact monitor on {MONITOR_URL}: {e}")
            return
        if response.status_code == 200:
            data = response.json()
            next_video = data.get("next_video")
            if not next_video:
                logging.error(f"No next video received for round 0")
                break
            break
        elif response.status_code == 500:
            logging.error(f"Preprocessing failed for round 0: {response.text}")
            break
        logging.info("Waiting for preprocessing to complete for round 0")
        time.sleep(5)

    # Intervention rounds
    for round_num in range(1, rounds + 1):
        add_action(puppet, f"round_{round_num}_start")
        
        # Fecth and dump the recommendations for this round
        recs = get_recommendations(puppet, focus, round_num)
        with open(os.path.join(puppet_shared_dir, f"recommendations_{round_num}.txt"), "w") as f:
            f.write("\n".join(recs))

        # Start preprocessing for this round
        try:
            response = requests.post(f"{MONITOR_URL}/start_preprocess", json={
                "puppet_id": puppet["puppetId"],
                "round_num": round_num,
                "intervention_type": intervention_type,
                "focus": focus
            }, timeout=30)
        except Exception as e:
            logging.error(f"Could not contact monitor on {MONITOR_URL}: {e}")
            return
        if response.status_code != 200:
            logging.error(f"Failed to start preprocess for round {round_num}: {response.text}")
            break

        # Wait for preprocessing to complete and get recommendations
        while True:
            try:
                response = requests.post(f"{MONITOR_URL}/get_recommendations", json={
                    "puppet_id": puppet["puppetId"],
                    "round_num": round_num,
                    "intervention_type": intervention_type,
                }, timeout=30)
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
            time.sleep(5)

        # Watch the video
        selected_video = Video(None, make_url(next_video))
        watch(puppet, selected_video, args["duration"])

        # Signal round completion
        response = requests.post(f"{MONITOR_URL}/complete_round", json={
            "puppet_id": puppet["puppetId"],
            "round_num": round_num
        }, timeout=30)
        if response.status_code != 200:
            logging.warning(f"Round {round_num} completion not acknowledged: {response.text}")
        logging.info(f"Signaled completion for round {round_num}")

        add_action(puppet, f"round_{round_num}_end")

    add_action(puppet, "intervention_end")

if __name__ == "__main__":
    args = json.loads(sys.argv[1])
    args["outputDir"] = args.get("outputDir", "/output")
    log_filename = f"{args['puppetId']}_{args.get('steps', 'unknown')}.log"
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=os.path.join('/logs', log_filename),
        level=logging.INFO,
        filemode='a'
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