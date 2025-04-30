from ytdriver import YTDriver, Video, VideoUnavailableException
import logging
import sys
import json
import time
from datetime import datetime
import os
# from random import choice
# import pandas as pd
import time
# import random

puppet = {}


def init_puppet(puppetId, profile_dir):
    """
    Creates a new puppet object with:
    1. A YouTube driver (browser automation)
    2. A unique ID
    3. Empty actions list to track what happens
    4. Start time to track duration
    """
    global puppet
    puppet = dict(
        driver=YTDriver(
            profile_dir=profile_dir, use_virtual_display=True
        ),
        puppetId=puppetId,
        actions=[],
        start_time=datetime.now(),
    )
    return puppet


def makedir(outputDir, d):
    dir = os.path.join(outputDir, d)
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def make_url(videoId):
    return "https://youtube.com/watch?v=" + str(videoId)


def add_action(action, params=None):
    # Records every action the puppet takes with timestamp
    logger.info(action, params)
    puppet["actions"].append(dict(action=action, params=params))


def get_homepage():
    homepage = puppet["driver"].get_homepage_recommendations(scroll_times=4)
    add_action("get_homepage_recommendations", [vid.videoId for vid in homepage])
    return homepage


def get_recommendations():
    recommendations = puppet["driver"].get_upnext_recommendations(topn=12)
    add_action("get_upnext_recommendations", [vid.videoId for vid in recommendations])
    return recommendations


def watch(video: Video, duration):
    driver = puppet["driver"]
    try:
        driver.play(video, duration=duration)
    except VideoUnavailableException as e:
        logger.info("Skipping unavailable video")
        # pass
    add_action("watch", video.videoId)


def save_puppet():
    js = dict(
        puppet_id=puppet["puppetId"],
        start_time=puppet["start_time"],
        end_time=datetime.now(),
        duration=puppet["duration"],
        description=puppet["description"],
        actions=puppet["actions"],
        args=args,
    )
    with open(os.path.join(makedir(args["outputDir"], "puppets"), puppet["puppetId"]), "w") as f:
        json.dump(js, f, default=str, indent=4)


def train():
    get_homepage()
    add_action("training_start")

    # Create screenshots directory
    screenshots_dir = os.path.join(args['outputDir'], 'screenshots', args['puppetId'])
    if not os.path.exists(screenshots_dir):
        os.makedirs(screenshots_dir)

    # get list of videoIds
    training = args["training"]
    logger.info("Training videos:\n%s", "\n".join(f"  {vid}" for vid in training))
    training_videos = [videoId for videoId in training if len(videoId) > 0]
    trainingN = int(args["trainingN"])
    # number of videos watched
    watched = 0

    last_video = None
    for videoId in training_videos:
        logger.info(f"Loading video: {videoId}")
        # watch until N videos have been watched
        if watched >= trainingN:
            break

        # Watch the video
        try:
            video = Video(None, make_url(videoId))
            watch(video, args["duration"])
            last_video = video
            watched += 1
        except VideoUnavailableException:
            continue
        except Exception as e:
            logger.exception(e)

    add_action("training_end")

    """ Get upnext recommendations after training """
    if last_video is not None:
        retry_upnext = 0
        # Take screenshot before getting recommendations
        puppet["driver"].save_screenshot(os.path.join(screenshots_dir, f"last_video_before_recs.png"))
        up_next = puppet["driver"].get_upnext_recommendations(topn=12)
        add_action("get_upnext_recommendations", [vid.videoId for vid in up_next])
    else:
        raise Exception("No video to get recommendations from {videoId}.")

    """ Get homepage recommendations after training """
    homepage = puppet["driver"].get_homepage_recommendations(scroll_times=4)
    # Take screenshot after first attempt
    puppet["driver"].save_screenshot(os.path.join(screenshots_dir, f"homepage_first_attempt.png"))
    add_action("get_homepage_recommendations", [vid.videoId for vid in homepage])



def intervention(initial_upnext=None, initial_homepage=None):
    try:
        if "intervention_type" in args:
            logger.info("Starting recommendation intervention experiment")
            add_action("intervention_start")
            from intervention import run_intervention
            run_intervention(puppet, args, initial_upnext=initial_upnext, initial_homepage=initial_homepage, logger=logger)
            logger.info("Recommendation intervention experiment completed")
    except Exception as e:
        logger.exception(f"Error in intervention step: {e}")
        raise

def extract_recommendations():
    upnext_recommendations = []
    homepage_recommendations = []
    
    # Extract the last occurrence of each recommendation type from actions
    for action in reversed(puppet["actions"]):
        if action["action"] == "get_upnext_recommendations" and not upnext_recommendations:
            upnext_recommendations = action["params"] or []
        elif action["action"] == "get_homepage_recommendations" and not homepage_recommendations:
            homepage_recommendations = action["params"] or []
        if upnext_recommendations and homepage_recommendations:
            break
    
    # Trim recommendations as per your requirement
    upnext_recommendations = upnext_recommendations[:12]
    homepage_recommendations = homepage_recommendations[:25]
    
    return upnext_recommendations, homepage_recommendations

if __name__ == "__main__":
    args = json.loads(sys.argv[1])
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', filename=f'/logs/{args["puppetId"]}', level=logging.INFO, filemode='w')
    logger = logging.getLogger(__name__)

    try:
        profile_dir = os.path.join(makedir(args["outputDir"], "profiles"), args["puppetId"])
        logger.info(f"Creating profile directory: {profile_dir}")
        logger.info("Successfully created profile directory")
        init_puppet(args["puppetId"], profile_dir)
        logger.info("Initialized sock puppet:", args["puppetId"])

        steps = args["steps"]
        logger.info(f"Executing step: {steps}")
        if steps == "train":
            train()
        elif steps == "intervention":
            intervention()
        elif steps == "combined":
            train()
            initial_upnext, initial_homepage = extract_recommendations()
            intervention(initial_upnext=initial_upnext, initial_homepage=initial_homepage)
        else:
            logger.error(f"Invalid step: {steps}")
            sys.exit(1)

        puppet["driver"].close()
        puppet["steps"] = args["steps"]
        puppet["duration"] = args["duration"]
        puppet["description"] = args["description"]
        save_puppet()
        logger.info('sock puppet finished')
    except Exception as e:
        logger.exception(e)