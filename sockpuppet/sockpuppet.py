from ytdriver import YTDriver, Video, VideoUnavailableException
import logging
import sys
import json
import time
from datetime import datetime
import os
from metadata_extractor import MetadataExtractor
from selenium.common.exceptions import TimeoutException, WebDriverException
import tempfile

def init_puppet(puppetId, profile_dir):
    puppet = dict(
        driver=YTDriver(profile_dir=profile_dir, use_virtual_display=True),
        puppetId=puppetId,
        actions=[],
        start_time=datetime.now(),
        rounds=[],
        harmful_exposure=[]
    )
    return puppet

def load_puppet(puppetId, args):
    puppet_file = os.path.join(args["outputDir"], "puppets", puppetId)
    if not os.path.exists(puppet_file):
        raise FileNotFoundError(f"Puppet file {puppet_file} not found")
    with open(puppet_file, "r") as f:
        puppet_data = json.load(f)
    profile_dir = os.path.join(args["outputDir"], "profiles", puppetId)
    puppet = dict(
        driver=YTDriver(profile_dir=profile_dir, use_virtual_display=True),
        puppetId=puppet_data["puppet_id"],
        actions=puppet_data["actions"],
        start_time=datetime.strptime(puppet_data["start_time"], "%Y-%m-%d %H:%M:%S.%f"),
        rounds=puppet_data.get("rounds", []),
        harmful_exposure=puppet_data.get("harmful_exposure", [])
    )
    return puppet

def makedir(outputDir, d):
    dir = os.path.join(outputDir, d)
    os.makedirs(dir, exist_ok=True)
    return dir

def make_url(videoId):
    return "https://youtube.com/watch?v=" + str(videoId)

def add_action(puppet, action, params=None):
    logging.info(f"Action: {action}, Params: {params}")
    puppet["actions"].append({
        "action": action,
        "params": params,
        "timestamp": datetime.now().isoformat()
    })

def get_homepage(puppet):
    homepage = puppet["driver"].get_homepage_recommendations(scroll_times=6)
    add_action(puppet, "get_homepage_recommendations", [vid.videoId for vid in homepage])
    return homepage

def get_recommendations(puppet):
    recommendations = puppet["driver"].get_upnext_recommendations(topn=12)
    add_action(puppet, "get_upnext_recommendations", [vid.videoId for vid in recommendations])
    return recommendations

def watch(puppet, video: Video, duration):
    driver = puppet["driver"]
    try:
        driver.play(video, duration=duration)
    except VideoUnavailableException as e:
        logging.info("Skipping unavailable video")
    add_action(puppet, "watch", video.videoId)

def save_puppet(puppet, args):
    js = dict(
        puppet_id=puppet["puppetId"],
        start_time=puppet["start_time"],
        end_time=datetime.now(),
        duration=puppet.get("duration", 0),
        description=puppet.get("description", ""),
        actions=puppet["actions"],
        rounds=puppet["rounds"],
        args=args,
        harmful_exposure=puppet.get("harmful_exposure", [])
    )
    puppet_file = os.path.join(makedir(args["outputDir"], "puppets"), puppet["puppetId"])
    with open(puppet_file, "w") as f:
        json.dump(js, f, default=str, indent=4)
    logging.info(f"Saved puppet state to {puppet_file}")

def extract_recommendations(puppet):
    logging.info(f"Extracting recommendations for puppet {puppet['puppetId']}")
    upnext_recommendations = []
    homepage_recommendations = []
    for action in reversed(puppet["actions"]):
        if action["action"] == "get_upnext_recommendations" and not upnext_recommendations:
            upnext_recommendations = action["params"] or []
        elif action["action"] == "get_homepage_recommendations" and not homepage_recommendations:
            homepage_recommendations = action["params"] or []
        if upnext_recommendations and homepage_recommendations:
            break
    upnext_recommendations = upnext_recommendations[:12]
    homepage_recommendations = homepage_recommendations[:25]
    return upnext_recommendations, homepage_recommendations

def save_experiment_log(metadata_dir, round_num, focus, video_ids, predictions=None, metadata_entries=None):
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
        "metadata": metadata_entries if metadata_entries is not None else [],
        "categories": [""] * len(video_ids) 
    }
    data["rounds"].append(round_data)
    
    with tempfile.NamedTemporaryFile(mode="w", dir=metadata_dir, delete=False) as tmp_file:
        json.dump(data, tmp_file, default=str, indent=4)
    os.replace(tmp_file.name, log_file)
    logging.info(f"Appended experiment log for round {round_num} to {log_file}")

def train(puppet, args):
    logging.info(f"Starting training for puppet {puppet['puppetId']}")
    get_homepage(puppet)
    add_action(puppet, "training_start")
    
    screenshots_dir = os.path.join(args['outputDir'], 'screenshots', args['puppetId'])
    os.makedirs(screenshots_dir, exist_ok=True)
    training = args.get("training", [])
    logging.info("Training videos received: %s", training)
    if not training:
        logging.warning("No training videos provided. Training phase will be skipped.")
        return
    
    training_videos = [videoId for videoId in training if len(videoId) > 0]
    trainingN = int(args["trainingN"])
    watched = 0
    last_video = None
    
    for videoId in training_videos:
        logging.info(f"Loading video: {videoId}")
        if watched >= trainingN:
            break
        for attempt in range(3):
            # logging.info(f"Attempt {attempt + 1} to watch video {videoId}")
            try:
                video = Video(None, make_url(videoId))
                watch(puppet, video, args["duration"])
                last_video = video
                watched += 1
                break
            except (VideoUnavailableException, TimeoutException, WebDriverException):
                logging.info(f"Attempt {attempt + 1} failed for video {videoId}")
                puppet["driver"].save_screenshot(os.path.join(screenshots_dir, f"error_{videoId}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
                if attempt == 2:
                    logging.info(f"Skipping unavailable video")
                    break
                time.sleep(10)
            except Exception as e:
                logging.exception(e)
                logging.info(f"Unrecoverable error for video {videoId}")
                break
                
    add_action(puppet, "training_end")
    
    experiment_dir = os.path.join(args['outputDir'], args['puppetId'])
    metadata_dir = os.path.join(experiment_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    metadata_extractor = MetadataExtractor(output_dir=metadata_dir, timeout=60, max_workers=4)
    if last_video is not None:
        retry_upnext = 0
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        puppet["driver"].save_screenshot(os.path.join(screenshots_dir, f"last_video_before_recs_{timestamp}.png"))
        up_next = puppet["driver"].get_upnext_recommendations(topn=12)
        add_action(puppet, "get_upnext_recommendations", [vid.videoId for vid in up_next])
        up_next_ids = [vid.videoId for vid in up_next]
        metadata = metadata_extractor.extract_metadata_batch(up_next_ids)
        save_experiment_log(metadata_dir, 0, "up-next", up_next_ids, [0.0] * len(up_next_ids), metadata)
    
    homepage = puppet["driver"].get_homepage_recommendations(scroll_times=6)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    puppet["driver"].save_screenshot(os.path.join(screenshots_dir, f"homepage_first_attempt_{timestamp}.png"))
    add_action(puppet, "get_homepage_recommendations", [vid.videoId for vid in homepage])
    homepage_ids = [vid.videoId for vid in homepage[:25]]
    metadata = metadata_extractor.extract_metadata_batch(homepage_ids)
    save_experiment_log(metadata_dir, 0, "homepage", homepage_ids, [0.0] * len(homepage_ids), metadata)
    logging.info(f"Completed training for puppet {puppet['puppetId']}")

def intervention(puppet, args, initial_upnext=None, initial_homepage=None):
    try:
        if "intervention_type" in args:
            logging.info("Starting recommendation intervention experiment")
            logging.info(f"Starting intervention for puppet {puppet['puppetId']}")
            add_action(puppet, "intervention_start")
            from intervention import run_intervention
            focus = args.get("focus", "homepage")
            run_intervention(args, puppet, logger=logging, initial_upnext=initial_upnext, initial_homepage=initial_homepage, focus=focus)
            logging.info("Recommendation intervention experiment completed")
    except Exception as e:
        logging.exception(f"Error in intervention step: {e}")
        raise

if __name__ == "__main__":
    args = json.loads(sys.argv[1])
    if "outputDir" not in args:
        args["outputDir"] = "/output"
    if args.get("steps") == "intervention":
        intervention_type = args.get("intervention_type", "unknown")
        focus_type = args.get("focus", "unknown")
        log_filename = f"{args['puppetId']}_intervention_{intervention_type}_{focus_type}.log"
    else:
        log_filename = f"{args['puppetId']}_{args.get('steps', 'unknown')}.log"
    
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', filename=os.path.join('/logs', log_filename), level=logging.INFO, filemode='a')
    logger = logging.getLogger(__name__)
    try:
        profile_dir = os.path.join(makedir(args["outputDir"], "profiles"), args["puppetId"])
        logger.info(f"Using profile directory: {profile_dir}")
        steps = args["steps"]
        logger.info(f"Executing step: {steps}")
          
        if steps == "train":
            puppet = init_puppet(args["puppetId"], profile_dir)
            logger.info(f"Initialized sock puppet: {args['puppetId']}")
            train(puppet, args)
        elif steps == "intervention":
            puppet = load_puppet(args["puppetId"], args)
            logger.info(f"Loaded sock puppet: {args['puppetId']}")
            initial_upnext, initial_homepage = extract_recommendations(puppet)
            intervention(puppet, args, initial_upnext=initial_upnext, initial_homepage=initial_homepage)
        elif steps == "combined":
            puppet = init_puppet(args["puppetId"], profile_dir)
            logger.info(f"Initialized sock puppet: {args['puppetId']}")
            train(puppet, args)
            initial_upnext, initial_homepage = extract_recommendations(puppet)
            intervention(puppet, args, initial_upnext=initial_upnext, initial_homepage=initial_homepage)
        else:
            logger.error(f"Invalid step: {steps}")
            sys.exit(1)
            
        puppet["steps"] = args["steps"]
        puppet["duration"] = args["duration"]
        puppet["description"] = args["description"]
        logger.info(f"Saving puppet state for {puppet['puppetId']}")
        save_puppet(puppet, args)
        puppet["driver"].close()
        logger.info('sock puppet finished')
    except Exception as e:
        logger.exception(e)