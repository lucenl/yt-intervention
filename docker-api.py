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

NUM_TRAINING_VIDEOS = 10
WATCH_DURATION = 5
USERNAME = os.getuid()

PERCENTAGE_GROUPS = [50]
PUPPETS_PER_GROUP = 5


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
    # New arguments for intervention mode
    parser.add_argument(
        "--intervention",
        action="store_true",
        help="Run intervention experiments instead of training",
    )  
    parser.add_argument(
        "--model-path",
        default="/app/models/roberta_checkpoint",
        help="Path to the RoBERTa model checkpoint inside container",
    )
    parser.add_argument(
            "--intervention-types",
            nargs="+",
            default=["none", "downrank", "replace"],
            choices=["none", "downrank", "replace"],
            help="Intervention types to run",
        )
    parser.add_argument(
    "--selection-type",
    default="decay_weighted_random",
    choices=["top", "random", "decay_weighted_random"],
    help="Video selection strategy for intervention"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of rounds per intervention experiment",
    )
    parser.add_argument(
        "--harmful-percentages",
        nargs="+",
        type=int,
        default=[50],
        help="Harmful percentage groups to use",
    )
    parser.add_argument(
        "--puppets-per-group",
        type=int,
        default=5,
        help="Number of puppets per group to use",
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
    return {OUTPUT_DIR: {"bind": "/output"}, LOGS_DIR: {"bind": "/logs"},
            os.path.abspath("roberta/checkpoint"): {"bind": "/app/models/roberta_checkpoint"}}



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


def validate_model_path(model_path):
    host_model_path = "roberta/checkpoint"  # Adjust to your host path
    if not os.path.exists(host_model_path):
        raise FileNotFoundError(f"Model directory {host_model_path} does not exist on host")
    print(f"Model directory {host_model_path} found on host")
    

def spawn_combined_containers(client, args):
    validate_model_path(args.model_path)
    harmful_pool, harmless_pool = load_video_pools(args)
    if harmful_pool is None or harmless_pool is None:
        raise FileNotFoundError("Failed to load video pools. Check the training-videos path and CSV files.")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)

    count = 0
    total_experiments = args.puppets_per_group * len(args.harmful_percentages) * len(args.intervention_types)

    print(f"Preparing to run {total_experiments} combined experiments")

    for percentage in args.harmful_percentages:
        training = get_training_videos(harmful_pool, harmless_pool, percentage)
        test_seed = training[-1]

        for puppet_idx in range(args.puppets_per_group):
            for intervention_type in args.intervention_types:
                while max_containers_reached(client, args.max_containers):
                    print("Max containers reached. Sleeping...")
                    sleep(args.sleep_duration)

                puppet_id = f"harmful_{percentage},{str(uuid4())[:8]}_{intervention_type}"
                profile_path = f"/output/profiles/{puppet_id}"

                experiment_args = {
                    "puppetId": puppet_id,
                    "profile_dir": profile_path,
                    "outputDir": "/output",
                    "duration": WATCH_DURATION,
                    "description": f"Combined experiment: {percentage}% harmful, {intervention_type}",
                    "steps": "combined",
                    "intervention_type": intervention_type,
                    "selection_type": args.selection_type,
                    "rounds": args.rounds,
                    "model_path": args.model_path,
                    "harm_threshold": 0.8,
                    "training": training,
                    "trainingN": NUM_TRAINING_VIDEOS,
                    "testSeed": test_seed
                }

                print(f"Starting experiment {count + 1}/{total_experiments}: {puppet_id}")
                if not args.simulate:
                    command = ["python", "sockpuppet.py", json.dumps(experiment_args)]
                    container = client.containers.run(
                        IMAGE_NAME,
                        command,
                        volumes=get_mount_volumes(),
                        shm_size="1G",
                        remove=True,
                        detach=True
                    )
                count += 1
                sleep(3)

    print(f"Launched {count} combined experiments")

def main():
    args, parser = parse_args()

    if args.build:
        print("Starting docker build...")
        build_image()
        print("Build complete!")

    if args.run:
        print("Starting docker containers...")
        client = docker.from_env()
        spawn_combined_containers(client, args)

    if not args.build and not args.run:
        parser.print_help()

if __name__ == "__main__":
    main()