from argparse import ArgumentParser
from random import choice
import docker
from time import sleep
import os
import pandas as pd
from uuid import uuid4
import json
import random
import shutil

# Change this to your own ID
IMAGE_NAME = "lucen/youtube-sock-puppet"
OUTPUT_DIR = os.path.join(os.getcwd(), "output")
LOGS_DIR = os.path.join(os.getcwd(), "logs")
ARGS_DIR = os.path.join(os.getcwd(), 'arguments')

NUM_TRAINING_VIDEOS = 5
WATCH_DURATION = 5
USERNAME = os.getuid()

PERCENTAGE_GROUPS = [50]
PUPPETS_PER_GROUP = 5
HARMLESS_RESERVOIR_SIZE = 10

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build docker image")
    parser.add_argument("--train-only", action="store_true", help="Run only the training phase")
    parser.add_argument("--intervention-only", action="store_true", help="Run only the intervention phase")
    parser.add_argument("--combined", action="store_true", help="Run both training and intervention phases")
    parser.add_argument(
        "--puppet-folder",
        default='output/puppets',
        help="Path to folder containing puppet state files for intervention (optional)"
    )
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
    parser.add_argument(
        "--binary-model-path",
        default="/app/models/roberta_checkpoint",
        help="Path to the binary RoBERTa model checkpoint inside container",
    )
    parser.add_argument(
        "--multiclass-model-path",
        default="/app/models/multicalss_checkpoint",
        help="Path to the multiclass RoBERTa model checkpoint inside container",
    )
    parser.add_argument(
        "--intervention-types",
        nargs="+",
        default=["downrank"],
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
        default=PERCENTAGE_GROUPS,
        help="Harmful percentage groups to use",
    )
    parser.add_argument(
        "--puppets-per-group",
        type=int,
        default=PUPPETS_PER_GROUP,
        help="Number of puppets per group to use",
    )
    parser.add_argument(
        "--focus",
        default="homepage",
        choices=["homepage", "up-next", "both"],
        help="Focus on recommendations (homepage, up-next, or both)"
    )
    parser.add_argument(
        "--dump",
        default=False,
        help="Dump recommendations to file",
    )
    
    args = parser.parse_args()
    return args, parser

def build_image():
    client = docker.from_env()
    client.images.build(path="./sockpuppet", tag=IMAGE_NAME, rm=True)

def get_mount_volumes():
    return {
        OUTPUT_DIR: {"bind": "/output"},
        LOGS_DIR: {"bind": "/logs"},
        os.path.abspath("roberta/binary"): {"bind": "/app/models/roberta_checkpoint"},
        os.path.abspath("roberta/multiclass"): {"bind": "/app/models/multicass_checkpoint"}
    }

def max_containers_reached(client, max_containers):
    try:
        return len(client.containers.list()) >= max_containers
    except:
        return True

def load_video_pools(args):
    try:
        TRAINING_BASE = args.training_videos
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
    # Sample from pools
    sampled_harmful = random.sample(harmful_pool["videoId"].tolist(), min(harmful_count, len(harmful_pool)))
    sampled_harmless = random.sample(harmless_pool["videoId"].tolist(), min(harmless_count, len(harmless_pool)))

    # Combine and shuffle
    combined_videos = sampled_harmful + sampled_harmless
    random.shuffle(combined_videos)
    
    return combined_videos

def sample_harmless_reservoir(harmless_pool, training_videos, size=HARMLESS_RESERVOIR_SIZE):
    available_harmless = [vid for vid in harmless_pool["videoId"].tolist() if vid not in training_videos]
    num_samples = min(size, len(available_harmless))
    if num_samples > 0:
        return random.sample(available_harmless, num_samples)
    return []

def validate_model_path(model_path):
    host_model_path = "roberta/checkpoint"
    if not os.path.exists(host_model_path):
        raise FileNotFoundError(f"Model directory {host_model_path} does not exist on host")
    print(f"Model directory {host_model_path} found on host")

def spawn_training_containers(client, args):
    harmful_pool, harmless_pool = load_video_pools(args)
    if harmful_pool is None or harmless_pool is None:
        raise FileNotFoundError("Failed to load video pools.")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
    
    count = 0
    total_experiments = args.puppets_per_group * len(args.harmful_percentages) * len(args.focus)
    print(f"Preparing to run {total_experiments} training experiments")
    for puppet_idx in range(args.puppets_per_group):
        for percentage in args.harmful_percentages:
            while max_containers_reached(client, args.max_containers):
                print("Max containers reached. Sleeping...")
                sleep(args.sleep_duration)
            training = get_training_videos(harmful_pool, harmless_pool, percentage)
            harmless_reservoir = sample_harmless_reservoir(harmless_pool, training)
            test_seed = training[-1] if training else str(uuid4())
            puppet_id = f"harmful_{percentage},{str(uuid4())[:8]}"
            profile_path = os.path.join("/output", "profiles", puppet_id)
            experiment_args = {
                "puppetId": puppet_id,
                "profile_dir": profile_path,
                "outputDir": "/output",
                "duration": WATCH_DURATION,
                "description": f"Training experiment: {percentage}% harmful",
                "steps": "train",
                "training": training,
                "trainingN": NUM_TRAINING_VIDEOS,
                "testSeed": test_seed,
                "harmless_reservoir": harmless_reservoir
            }
            print(f"Starting training experiment {count + 1}/{total_experiments}: {puppet_id}_train")
            if not args.simulate:
                command = ["python", "sockpuppet.py", json.dumps(experiment_args)]
                container = client.containers.run(
                    IMAGE_NAME, command, volumes=get_mount_volumes(), shm_size="1G", remove=True, detach=True
                )
            count += 1
            sleep(3)
    print(f"Launched {count} training experiments")

def spawn_intervention_containers(client, args):
    validate_model_path(args.model_path)
    harmful_pool, harmless_pool = load_video_pools(args)
    if harmful_pool is None or harmless_pool is None:
        raise FileNotFoundError("Failed to load video pools.")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
    count = 0
    puppet_ids = []
    if args.puppet_folder:
        puppet_folder = args.puppet_folder
        if not os.path.exists(puppet_folder):
            raise FileNotFoundError(f"Puppet folder {puppet_folder} does not exist")
        puppet_files = [f for f in os.listdir(puppet_folder) if os.path.isfile(os.path.join(puppet_folder, f))]
        puppet_ids = [os.path.splitext(f)[0] for f in puppet_files]
        total_experiments = len(puppet_ids) * len(args.intervention_types)
        print(f"Found {len(puppet_ids)} puppets in folder {puppet_folder}")
   
    for idx, intervention_type in enumerate(args.intervention_types):
        for focux_idx, focus_type in enumerate(args.focus):
            if args.puppet_folder:
                for puppet_idx, puppet_id in enumerate(puppet_ids):
                    while max_containers_reached(client, args.max_containers):
                        print("Max containers reached. Sleeping...")
                        sleep(args.sleep_duration)
                    puppet_file = os.path.join(args.puppet_folder, puppet_id)
                    with open(puppet_file, "r") as f:
                        puppet_data = json.load(f)
                    training_videos = puppet_data.get("args", {}).get("training", [])
                    harmless_reservoir = puppet_data.get("args", {}).get("harmless_reservoir", [])
                    orig_profile_path = os.path.join("/output", "profiles", puppet_id)
                    new_profile_path = os.path.join("/output", "profiles", f"{puppet_id}_intervention_{intervention_type}_{focus_type}")
                    if os.path.exists(orig_profile_path):
                        try:
                            shutil.copytree(orig_profile_path, new_profile_path, dirs_exist_ok=True)
                        except Exception as e:
                            print(f"Error copying profile: {e}")
                            continue
                        else:
                            num_files = len(os.listdir(new_profile_path))
                            print(f"Copied {num_files} files from {orig_profile_path} to {new_profile_path}")
                    experiment_args = {
                        "puppetId": puppet_id,
                        "profile_dir": new_profile_path,
                        "outputDir": "/output",
                        "duration": puppet_data.get("args", {}).get("duration", WATCH_DURATION),
                        "description": f"Intervention experiment: {puppet_id}, {intervention_type}, {focus_type}",
                        "steps": "intervention",
                        "intervention_type": intervention_type,
                        "selection_type": args.selection_type,
                        "rounds": args.rounds,
                        "model_path": args.model_path,
                        "harm_threshold": 0.8,
                        "focus": focus_type,
                        "training": training_videos,
                        "trainingN": puppet_data.get("args", {}).get("trainingN", NUM_TRAINING_VIDEOS),
                        "testSeed": puppet_data.get("args", {}).get("testSeed", str(uuid4())),
                        "harmless_reservoir": harmless_reservoir,
                        "dump": args.dump
                    }
                    print(f"Starting intervention experiment {count + 1}/{total_experiments}: {puppet_id}_intervention_{intervention_type}_{focus_type}")
                    if not args.simulate:
                        command = ["python", "sockpuppet.py", json.dumps(experiment_args)]
                        container = client.containers.run(
                            IMAGE_NAME, command, volumes=get_mount_volumes(), shm_size="1G", remove=True, detach=True
                        )
                    count += 1
                    sleep(5)
            else:
                raise FileNotFoundError(f'No puppet folder found. Please train puppets first before for intervention')
    
    print(f"Launched {count} intervention experiments")

def spawn_combined_containers(client, args):
    validate_model_path(args.model_path)
    harmful_pool, harmless_pool = load_video_pools(args)
    if harmful_pool is None or harmless_pool is None:
        raise FileNotFoundError("Failed to load video pools.")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
    count = 0
    total_experiments = args.puppets_per_group * len(args.harmful_percentages) * len(args.intervention_types)
    print(f"Preparing to run {total_experiments} combined experiments")
    for percentage in args.harmful_percentages:
        for puppet_idx in range(args.puppets_per_group):
            training = get_training_videos(harmful_pool, harmless_pool, percentage)
            harmless_reservoir = sample_harmless_reservoir(harmless_pool, training)
            test_seed = training[-1] if training else str(uuid4())
            for intervention_type in args.intervention_types:
                while max_containers_reached(client, args.max_containers):
                    print("Max containers reached. Sleeping...")
                    sleep(args.sleep_duration)
                puppet_id = f"harmful_{percentage},{str(uuid4())[:8]}"
                profile_path = os.path.join("/output", "profiles", puppet_id)
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
                    "focus": args.focus,
                    "training": training,
                    "trainingN": NUM_TRAINING_VIDEOS,
                    "testSeed": test_seed,
                    "harmless_reservoir": harmless_reservoir,
                    "dump": args.dump
                }
                print(f"Starting combined experiment {count + 1}/{total_experiments}: {puppet_id}_combined_{intervention_type}")
                if not args.simulate:
                    command = ["python", "sockpuppet.py", json.dumps(experiment_args)]
                    container = client.containers.run(
                        IMAGE_NAME, command, volumes=get_mount_volumes(), shm_size="1G", remove=True, detach=True
                    )
                count += 1
                sleep(5)
    print(f"Launched {count} combined experiments")

def main():
    args, parser = parse_args()
    if args.build:
        print("Starting docker build...")
        build_image()
        print("Build complete!")
    client = docker.from_env()
    if args.train_only:
        print("Starting training containers...")
        spawn_training_containers(client, args)
    elif args.intervention_only:
        print("Starting intervention containers...")
        spawn_intervention_containers(client, args)
    elif args.combined:
        print("Starting combined containers...")
        spawn_combined_containers(client, args)
    if not any([args.build, args.train_only, args.intervention_only, args.combined]):
        parser.print_help()

if __name__ == "__main__":
    main()