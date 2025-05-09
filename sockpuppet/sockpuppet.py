from ytdriver import YTDriver, Video, VideoUnavailableException
import logging
import sys
import json
import time
from datetime import datetime
import os

def init_puppet(puppetId, profile_dir):
    """
    Creates a new puppet object with:
    1. A YouTube driver (browser automation)
    2. A unique ID
    3. Empty actions list to track what happens
    4. Start time to track duration
    5. Empty rounds list to track intervention rounds
    6. Empty harmful_exposure list to track harmful content exposure
    """
    puppet = dict(
        driver=YTDriver(
            profile_dir=profile_dir, use_virtual_display=True
        ),
        puppetId=puppetId,
        actions=[],
        start_time=datetime.now(),
        rounds=[],
        harmful_exposure=[]
    )
    return puppet

def load_puppet(puppetId, args):
    """
    Load a saved puppet state from file.
    """
    puppet_file = os.path.join(args["outputDir"], "puppets", puppetId)
    if not os.path.exists(puppet_file):
        raise FileNotFoundError(f"Puppet file {puppet_file} not found")
    
    with open(puppet_file, "r") as f:
        puppet_data = json.load(f)
    
    # Reinitialize the driver
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
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def make_url(videoId):
    return "https://youtube.com/watch?v=" + str(videoId)

def add_action(puppet, action, params=None):
    """
    Add an action to the puppet's action log
    
    Args:
        puppet: Puppet dictionary
        action: Name of the action
        params: Optional parameters for the action
    """
    logger.info(f"Action: {action}, Params: {params}")
    puppet["actions"].append({
        "action": action,
        "params": params,
        "timestamp": datetime.now().isoformat()
    })

def get_homepage(puppet):
    homepage = puppet["driver"].get_homepage_recommendations(scroll_times=4)
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
        logger.info("Skipping unavailable video")
    add_action(puppet, "watch", video.videoId)

def save_puppet(puppet, args):
    js = dict(
        puppet_id=puppet["puppetId"],
        start_time=puppet["start_time"],
        end_time=datetime.now(),
        duration=puppet["duration"],
        description=puppet["description"],
        actions=puppet["actions"],
        rounds=puppet["rounds"],
        args=args,
        harmful_exposure=puppet.get("harmful_exposure", [])
    )
    with open(os.path.join(makedir(args["outputDir"], "puppets"), puppet["puppetId"]), "w") as f:
        json.dump(js, f, default=str, indent=4)

def train(puppet, args):
    logger.info(f"Puppet state at start of train: {puppet}")
    get_homepage(puppet)
    add_action(puppet, "training_start")

    screenshots_dir = os.path.join(args['outputDir'], 'screenshots', args['puppetId'])
    if not os.path.exists(screenshots_dir):
        os.makedirs(screenshots_dir)

    training = args.get("training", [])
    logger.info("Training videos received: %s", training)
    if not training:
        logger.warning("No training videos provided. Training phase will be skipped.")
        return

    training_videos = [videoId for videoId in training if len(videoId) > 0]
    trainingN = int(args["trainingN"])
    watched = 0
    last_video = None

    for videoId in training_videos:
        logger.info(f"Loading video: {videoId}")
        if watched >= trainingN:
            break
        try:
            video = Video(None, make_url(videoId))
            watch(puppet, video, args["duration"])
            last_video = video
            watched += 1
        except VideoUnavailableException:
            continue
        except Exception as e:
            logger.exception(e)

    add_action(puppet, "training_end")

    if last_video is not None:
        retry_upnext = 0
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        puppet["driver"].save_screenshot(os.path.join(screenshots_dir, f"last_video_before_recs_{timestamp}.png"))
        up_next = puppet["driver"].get_upnext_recommendations(topn=12)
        add_action(puppet, "get_upnext_recommendations", [vid.videoId for vid in up_next])
    else:
        raise Exception("No video to get recommendations from.")

    homepage = puppet["driver"].get_homepage_recommendations(scroll_times=4)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    puppet["driver"].save_screenshot(os.path.join(screenshots_dir, f"homepage_first_attempt_{timestamp}.png"))
    add_action(puppet, "get_homepage_recommendations", [vid.videoId for vid in homepage])
    logger.info(f"Puppet state at end of train: {puppet}")

def intervention(puppet, args, initial_upnext=None, initial_homepage=None):
    try:
        if "intervention_type" in args:
            logger.info("Starting recommendation intervention experiment")
            logger.info(f"Puppet state before intervention start: {puppet}")
            add_action(puppet, "intervention_start")
            from intervention import run_intervention
            logger.info(f"Puppet state before run_intervention: {puppet}")
            focus = args.get("focus", "homepage")
            run_intervention(args, puppet, logger=logger, initial_upnext=initial_upnext, initial_homepage=initial_homepage, focus=focus)
            logger.info("Recommendation intervention experiment completed")
    except Exception as e:
        logger.exception(f"Error in intervention step: {e}")
        raise

def extract_recommendations(puppet):
    logger.info(f"Puppet state in extract_recommendations: {puppet}")
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

if __name__ == "__main__":
    args = json.loads(sys.argv[1])
    if "outputDir" not in args:
        args["outputDir"] = "/output"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', filename=f'/logs/{args["puppetId"]}_{timestamp}.log', level=logging.INFO, filemode='w')
    logger = logging.getLogger(__name__)

    try:
        profile_dir = os.path.join(makedir(args["outputDir"], "profiles"), args["puppetId"])
        logger.info(f"Creating profile directory: {profile_dir}")
        logger.info("Successfully created profile directory")

        steps = args["steps"]
        logger.info(f"Executing step: {steps}")

        if steps == "train":
            puppet = init_puppet(args["puppetId"], profile_dir)
            logger.info("Initialized sock puppet: %s", args["puppetId"])
            logger.info(f"Puppet state after init: {puppet}")
            train(puppet, args)
        elif steps == "intervention":
            puppet = load_puppet(args["puppetId"], args)
            logger.info("Loaded sock puppet: %s", args["puppetId"])
            logger.info(f"Puppet state after load: {puppet}")
            initial_upnext, initial_homepage = extract_recommendations(puppet)
            intervention(puppet, args, initial_upnext=initial_upnext, initial_homepage=initial_homepage)
        elif steps == "combined":
            puppet = init_puppet(args["puppetId"], profile_dir)
            logger.info("Initialized sock puppet: %s", args["puppetId"])
            logger.info(f"Puppet state after init: {puppet}")
            train(puppet, args)
            initial_upnext, initial_homepage = extract_recommendations(puppet)
            intervention(puppet, args, initial_upnext=initial_upnext, initial_homepage=initial_homepage)
        else:
            logger.error(f"Invalid step: {steps}")
            sys.exit(1)

        puppet["steps"] = args["steps"]
        puppet["duration"] = args["duration"]
        puppet["description"] = args["description"]
        logger.info(f"Puppet state before save: {puppet}")
        save_puppet(puppet, args)
        puppet["driver"].close()
        logger.info('sock puppet finished')
    except Exception as e:
        logger.exception(e)