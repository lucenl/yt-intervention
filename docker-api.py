from argparse import ArgumentParser
import docker
from time import sleep
import os
import pandas as pd
from uuid import uuid4
import json
import random

IMAGE_NAME = "lucen/youtube-sock-puppet"
OUTPUT_DIR = os.path.join(os.getcwd(), "output")
LOGS_DIR = os.path.join(os.getcwd(), "logs")
SHARED_DIR = os.path.join(os.getcwd(), "shared")

NUM_TRAINING_VIDEOS = 5
WATCH_DURATION = 5

PERCENTAGE_GROUPS = [50]
PUPPETS_PER_GROUP = 2

ROUNDS = 10

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build docker image")
    parser.add_argument("--run", action="store_true", help="Run all docker containers")
    parser.add_argument("--simulate", action="store_true", help="Only generate arguments but do not start containers")
    parser.add_argument("--max-containers", default=10, type=int, help="Maximum number of concurrent containers")
    parser.add_argument("--sleep-duration", default=60, type=int, help="Time to sleep when max containers are reached")
    parser.add_argument("--training-videos", default="data/training/", help="Path to the training videos folder")
    parser.add_argument("--steps", default="combined", choices=["train", "intervention", "combined"], help="Steps to perform")
    parser.add_argument("--puppets-per-group", type=int, default=PUPPETS_PER_GROUP, help="Number of puppets per group")
    parser.add_argument("--harmful-percentages", type=float, nargs="+", default=PERCENTAGE_GROUPS, help="Percentage of harmful videos")
    parser.add_argument("--focus", type=str, default="homepage", choices=["homepage", "upnext"], help="Focus for recommendations")
    return parser.parse_args()

def build_image():
    client = docker.from_env()
    client.images.build(path='./sockpuppet', tag=IMAGE_NAME, rm=True)
   

def get_mount_volumes():
    return {
        OUTPUT_DIR: {"bind": "/output"},
        LOGS_DIR: {"bind": "/logs"},
        SHARED_DIR: {"bind": "/shared"}
    }

def max_containers_reached(client, max_containers):
    try:
        return len(client.containers.list()) >= max_containers
    except:
        return True

def load_video_pools(args):
    TRAINING_BASE = args.training_videos
    try:
        harmful_pool = pd.read_csv(os.path.join(TRAINING_BASE, "harmful.csv"))
        non_harmful_pool = pd.read_csv(os.path.join(TRAINING_BASE, "non_harmful.csv"))
        print(f"Loaded {len(harmful_pool)} harmful videos and {len(non_harmful_pool)} non-harmful videos.")
        return harmful_pool, non_harmful_pool
    except Exception as e:
        print(f"Error loading video pools: {e}")
        return None, None

def get_training_videos(harmful_pool, harmless_pool, harmful_percentage):
    harmful_count = int(NUM_TRAINING_VIDEOS * (harmful_percentage / 100))
    harmless_count = NUM_TRAINING_VIDEOS - harmful_count

    sampled_harmful = random.sample(harmful_pool["videoId"].tolist(), min(harmful_count, len(harmful_pool)))
    sampled_harmless = random.sample(harmless_pool["videoId"].tolist(), min(harmless_count, len(harmless_pool)))

    combined_videos = sampled_harmful + sampled_harmless
    random.shuffle(combined_videos)
    return combined_videos

def spawn_containers(args):
    client = docker.from_env()

    harmful_pool, harmless_pool = load_video_pools(args)
    if harmful_pool is None or harmless_pool is None:
        print("No video pools available. Exiting.")
        exit(1)

    for d in [OUTPUT_DIR, LOGS_DIR, SHARED_DIR, os.path.join(OUTPUT_DIR, "profiles"), os.path.join(OUTPUT_DIR, "puppets")]:
        os.makedirs(d, exist_ok=True)
        os.chmod(d, 0o777)

    count = 0
    total_experiments = args.puppets_per_group * len(args.harmful_percentages)
    print(f"Preparing to run {total_experiments} experiments with steps={args.steps}")
    for puppet_idx in range(args.puppets_per_group):
        for percentage in args.harmful_percentages:
            while max_containers_reached(client, args.max_containers):
                print("Max containers reached. Sleeping...")
                sleep(args.sleep_duration)

            training = get_training_videos(harmful_pool, harmless_pool, percentage)
            puppet_id = f"harmful_{percentage}_{str(uuid4())[:8]}"
            puppet_args = {
                "puppetId": puppet_id,
                "duration": WATCH_DURATION,
                "description": f"Sock puppet with {percentage}% harmful videos",
                "harmful_percentage": percentage,
                "outputDir": "/output",
                "training": training,
                "trainingN": NUM_TRAINING_VIDEOS,
                "steps": args.steps,
                "rounds": ROUNDS,
                "focus": args.focus
            }

            if not args.simulate:
                print(f"Spawning puppet {puppet_args['puppetId']} with steps {args.steps}...")
                command = ["python", "sockpuppet.py", json.dumps(puppet_args)]
                client.containers.run(
                    IMAGE_NAME, command, volumes=get_mount_volumes(), shm_size="512M", remove=True, detach=True
                )

            count += 1
            sleep(3)
    print(f"Total containers spawned: {count}")

def main():
    args = parse_args()

    if args.build:
        build_image()

    if args.run:
        spawn_containers(args)

    if not args.build and not args.run:
        print("No operation specified. Use --build or --run.")

if __name__ == "__main__":
    main()