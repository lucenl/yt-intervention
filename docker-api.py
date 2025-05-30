from argparse import ArgumentParser
import docker
from time import sleep
import os
import pandas as pd
from uuid import uuid4
import json
import random
import logging
import stat
import shutil

IMAGE_NAME = "lucen/youtube-sock-puppet"

OUTPUT_DIR_PATH = "debug-output"
LOGS_DIR_PATH = "logs"
SHARED_DIR_PATH = "debug-shared"

OUTPUT_DIR = os.path.join(os.getcwd(), OUTPUT_DIR_PATH)
LOGS_DIR = os.path.join(os.getcwd(), LOGS_DIR_PATH)
SHARED_DIR = os.path.join(os.getcwd(), SHARED_DIR_PATH)

DOCKER_OUTPUT_DIR = f"/{OUTPUT_DIR_PATH}"
DOCKER_LOGS_DIR = f"/{LOGS_DIR_PATH}"
DOCKER_SHARED_DIR = f"/{SHARED_DIR_PATH}"

print(f"os.getcwd(): {os.getcwd()}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Logs directory: {LOGS_DIR}")
print(f"Shared directory: {SHARED_DIR}")

NUM_TRAINING_VIDEOS = 5
WATCH_DURATION = 5

PERCENTAGE_GROUPS = [0, 50]
PUPPETS_PER_GROUP = 1

ROUNDS = 3

# Setup logging
logging.basicConfig(
    filename="docker-api.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)
logger = logging.getLogger(__name__)

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
    parser.add_argument("--harmful-percentages", type=int, nargs="+", default=PERCENTAGE_GROUPS, help="Percentage of harmful videos")
    parser.add_argument("--focus", type=str, default="homepage", choices=["homepage", "upnext"], help="Focus for recommendations")
    parser.add_argument("--puppet-id", type=str, help="Puppet ID (required for intervention, optional for train)")
    parser.add_argument("--harmful-percentage", type=int, default=0, help="Percentage of harmful videos")
    parser.add_argument("--intervention-type", type=str, choices=["none", "downrank", "replace"], default="none", help="Intervention type")
    parser.add_argument('--network', type=str, default='bridge', help='Docker network mode')
    return parser.parse_args()

def build_image():
    client = docker.from_env()
    print("Building Docker image...")
    try:
        client.images.build(path='./sockpuppet', tag=IMAGE_NAME, rm=True)
        print("Docker image built successfully")
    except Exception as e:
        print(f"Failed to build Docker image: {str(e)}")
        raise

def get_mount_volumes():
    return {
        # OUTPUT_DIR: {"bind": "/debug-output"},
        # SHARED_DIR: {"bind": "/debug-shared"},
        OUTPUT_DIR: {"bind": DOCKER_OUTPUT_DIR},
        SHARED_DIR: {"bind": DOCKER_SHARED_DIR},
        LOGS_DIR: {"bind": DOCKER_LOGS_DIR},
    }

print(f"Mounting volumes: {get_mount_volumes()}")

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
        logger.info(f"Loaded {len(harmful_pool)} harmful videos and {len(non_harmful_pool)} non-harmful videos.")
        return harmful_pool, non_harmful_pool
    except Exception as e:
        logger.error(f"Error loading video pools: {e}")
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
        logger.error("No video pools available. Exiting.")
        exit(1)

    count = 0
    total_experiments = args.puppets_per_group * len(args.harmful_percentages)
    print(f"Preparing to run {total_experiments} experiments with steps={args.steps}")
    for puppet_idx in range(args.puppets_per_group): # 30
        for percentage in args.harmful_percentages: # [0. 25. 50]
            while max_containers_reached(client, args.max_containers):
                logger.info("Max containers reached. Sleeping...")
                sleep(args.sleep_duration)

            training = get_training_videos(harmful_pool, harmless_pool, percentage)
            # puppet_id = f"harmful_{percentage},{str(uuid4())[:8]}"
            
            if args.puppet_id:
                puppet_id = args.puppet_id
                print(f"* Using provided puppet_id: {puppet_id}")
            else:
                puppet_id = f"harmful_{percentage},{str(uuid4())[:8]}"
                print(f"* Generated puppet_id: {puppet_id}")
            # Assign intervention type: alternate between "replace" and "downrank"
            intervention_types = ["replace", "downrank", "none"]
            intervention_type = intervention_types[puppet_idx % len(intervention_types)]
            print(f"* Using intervention type: {intervention_type}")
            print(f"* Output directory: {OUTPUT_DIR}; docker output dir: {DOCKER_OUTPUT_DIR}")
            print(f"* Log directory: {LOGS_DIR}; docker log dir: {DOCKER_LOGS_DIR}")
            puppet_args = {
                "puppetId": puppet_id,
                "duration": WATCH_DURATION,
                "description": f"Sock puppet with {percentage}% harmful videos",
                "harmful_percentage": percentage,
                "outputDir": DOCKER_OUTPUT_DIR,
                "training": training,
                "trainingN": NUM_TRAINING_VIDEOS,
                "steps": args.steps,
                "rounds": ROUNDS,
                "focus": args.focus,
                "intervention_type": intervention_type  # Add this
            }
            
            host_shared = os.path.join(os.getcwd(), SHARED_DIR_PATH)
            puppet_shared_host = os.path.join(host_shared, puppet_id)
            os.makedirs(puppet_shared_host, exist_ok=True)
            
            if not args.simulate:
                print(puppet_args["puppetId"])
                logger.info(f"Spawning puppet {puppet_args['puppetId']} with steps {args.steps}, intervention_type {intervention_type}...")
                command = ["python", "sockpuppet.py", json.dumps(puppet_args)]
                try:
                    container = client.containers.run(
                        IMAGE_NAME,
                        command,
                        volumes=get_mount_volumes(),
                        shm_size="512M",
                        remove=True,
                        detach=True,
                        network=args.network,
                        extra_hosts={"host.docker.internal": "host-gateway"}
                    )
                    logger.info(f"Started container {container.id} for puppet {puppet_id}")
                except Exception as e:
                    logger.error(f"Failed to start container for puppet {puppet_id}: {str(e)}")
                    
            count += 1
            sleep(3)
    logger.info(f"Total containers spawned: {count}")


def main():
    args = parse_args()

    if args.build:
        build_image()

    if args.run:
        spawn_containers(args)

    if not args.build and not args.run:
        logger.info("No operation specified. Use --build or --run.")

if __name__ == "__main__":
    main()