from argparse import ArgumentParser
from random import choice
import docker
from time import sleep
import os
import pandas as pd
from uuid import uuid4
import json
import random

# change this to your own ID
IMAGE_NAME = "lucen/youtube-sock-puppet"
OUTPUT_DIR = os.path.join(os.getcwd(), "output")
LOGS_DIR = os.path.join(os.getcwd(), "logs")
ARGS_DIR = os.path.join(os.getcwd(), 'arguments')

NUM_TRAINING_VIDEOS = 100
WATCH_DURATION = 30
USERNAME = os.getuid()

PERCENTAGE_GROUPS = [5]
PUPPETS_PER_GROUP = 10


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build docker image")
    parser.add_argument("--run", action="store_true", help="Run all docker containers")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Only generate arguments but do not start containers",
    )
    parser.add_argument(
        "--max-containers",
        default=10,
        type=int,
        help="Maximum number of concurrent containers",
    )
    parser.add_argument(
        "--sleep-duration",
        default=60,
        type=int,
        help="Time to sleep (in seconds) when max containers are reached and before spawning additional containers",
    )
    parser.add_argument(
        "--training-videos",
        default="data/training/",
        help="Path to the training videos folder",
    )

    args = parser.parse_args()
    return args, parser


def build_image():
    # get docker client and build image
    client = docker.from_env()

    # build the image from the Dockerfile
    client.images.build(path="./sockpuppet", tag=IMAGE_NAME, rm=True)
    

def get_mount_volumes():
    # binds "/output" on the container -> "OUTPUT_DIR" actual folder on disk
    # binds "/args" on the container -> "ARGS_DIR" actual folder on disk
    return {OUTPUT_DIR: {"bind": "/output"}, LOGS_DIR: {"bind": "/logs"}}


def max_containers_reached(client, max_containers):
    try:
        return len(client.containers.list()) >= max_containers
    except:
        return True

def load_video_pools(args):
    # Load harmful and non-harmful videos from csv files
    try:
        TRAINING_BASE = args.training_videos
        harmful_pool = pd.read_csv(os.path.join(TRAINING_BASE, "harmful.csv"))
        non_harmful_pool = pd.read_csv(os.path.join(TRAINING_BASE, "non_harmful.csv"))
        print(
            f"Loaded {len(harmful_pool)} harmful videos and {len(non_harmful_pool)} non-harmful videos."
        )
        return harmful_pool, non_harmful_pool
    except Exception as e:
        print(f"Error loading video pools: {e}")
        return None, None

def get_training_videos(harmful_pool, harmless_pool, harmful_percentage):
    """
    Prepare a training set with specific harmful content percentage

    Args:
        harmful_pool: List of harmful video IDs
        harmless_pool: List of harmless video IDs
        harmful_percentage: Target percentage of harmful videos (0-100)

    Returns:
        List of video IDs with specified distribution and shuffled order
    """
    harmful_count = int(NUM_TRAINING_VIDEOS * (harmful_percentage / 100))
    harmless_count = int(NUM_TRAINING_VIDEOS * ((100 - harmful_percentage) / 100))

    # Sample from pools
    sampled_harmful = random.sample(harmful_pool["videoId"].tolist(), harmful_count)
    sampled_harmless = random.sample(harmless_pool["videoId"].tolist(), harmless_count)

    # print(f"Harmful videos selected: {sampled_harmful}")
    # print(f"Non-harmful videos selected: {sampled_harmless}")
    # Combine and shuffle
    combined_videos = sampled_harmful + sampled_harmless
    random.shuffle(combined_videos)
    training_videos = combined_videos

    return training_videos


def spawn_containers(args):
    # get docker client
    client = docker.from_env()
    # Load video pools
    harmful_pool, harmless_pool = load_video_pools(args)

    # create required directories: makes sure we have directories for storing:
    # LOGS_DIR: for storing logs from each puppet
    # OUTPUT_DIR: for storing results from each puppet

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)

    # spawn containers for each user
    count = 0

    # percentage = PERCENTAGE_GROUPS[1] # 50% harmful
    while count < PUPPETS_PER_GROUP * len(PERCENTAGE_GROUPS):
        for percentage in PERCENTAGE_GROUPS:
            # for puppetIdx in range(PUPPETS_PER_GROUP):
            # check for running container list
            while max_containers_reached(client, args.max_containers):
                # sleep for a minute if maxContainers are active
                print("Max containers reached. Sleeping...")
                sleep(args.sleep_duration)

            training = get_training_videos(harmful_pool, harmless_pool, percentage)
            
            #TODO: we will use random one from top upnext as test seed
            # Set test seed as the last video in the training set
            # But actually it will be the REAL last video wathched by the puppet
            testSeed = training[-1]
            # Or it should be random?
            # testSeed = choice(training_videos)

            # generate a unique puppet identifier
            puppetId = f"harmful_{percentage},{str(uuid4())[:8]}"

            # write arguments to a file
            puppetArgs = dict(
                puppetId=puppetId,
                # duration to watch each video
                duration=WATCH_DURATION,
                # a description
                description=f"Train sock puppets with {percentage}% harmful videos",
                harmful_percentage=percentage,
                # output directory for sock puppet
                outputDir="/output",
                # videos to watch
                training=training,
                # number of training videos
                trainingN=NUM_TRAINING_VIDEOS,
                # intervention=videos,
                # seed video
                testSeed=testSeed,
                # steps to perform
                steps="train",  # Specify the steps to perform
            )
            
            # with open(os.path.join(ARGS_DIR, f'{puppetId}.json'), 'w') as f:
            #     json.dump(puppetArgs, f, indent=4)
                
            # spawn container if it's not a simulation
            if not args.simulate:
                print(f"Spawning puppet {puppetArgs['puppetId']}...")
                # set outputDir as "/output"
                command = ["python", "sockpuppet.py", json.dumps(puppetArgs)]

                # run the container
                # TODO: removed user
                container = client.containers.run(IMAGE_NAME, command, volumes=get_mount_volumes(), shm_size="512M", remove=True, detach=True)

            # increment count of containers
            count += 1
        print("Total containers spawned:", count)


def main():

    args, parser = parse_args()

    if args.build:
        print("Starting docker build...")
        build_image()
        print("Build complete!")

    if args.run:
        print("Starting docker containers...")
        spawn_containers(args)

    if not args.build and not args.run:
        parser.print_help()


if __name__ == "__main__":
    main()