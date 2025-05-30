#!/usr/bin/env python3
"""
YouTube Sock Puppet Orchestrator
A modular system for managing two-stage YouTube recommendation experiments:
1. Training stage: Create puppets with different harmful content percentages
2. Intervention stage: Test intervention strategies on trained puppets
"""

from argparse import ArgumentParser
import docker
from time import sleep
import os
import pandas as pd
from uuid import uuid4
import json
import random
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import threading
from queue import Queue
import concurrent.futures

# Configuration Constants
IMAGE_NAME = "lucen/youtube-sock-puppet"
NUM_TRAINING_VIDEOS = 110
WATCH_DURATION = 30
ROUNDS = 30

# Directory name constants
OUTPUT_DIR_NAME = "debug-output"
LOGS_DIR_NAME = "logs"
SHARED_DIR_NAME = "debug-shared"
PROFILES_DIR_NAME = "profiles"
PUPPETS_DIR_NAME = "puppets"

# Docker mount path constants
DOCKER_OUTPUT_DIR_PREFIX = "debug-output"
DOCKER_LOGS_DIR_PREFIX = "logs"
DOCKER_SHARED_DIR_PREFIX = "debug-shared"

class StageType(Enum):
    TRAIN = "train"
    INTERVENTION = "intervention"

class InterventionType(Enum):
    NONE = "none"
    DOWNRANK = "downrank"
    REPLACE = "replace"

class FocusType(Enum):
    HOMEPAGE = "homepage"
    UPNEXT = "upnext"

@dataclass
class PuppetConfig:
    """Configuration for a single puppet"""
    puppet_id: str
    stage: StageType
    harmful_percentage: int
    focus: Optional[FocusType] = None
    intervention_type: Optional[InterventionType] = None
    base_puppet_id: Optional[str] = None  # For intervention stage
    session_id: str = ""

@dataclass
class SessionConfig:
    """Configuration for a complete experimental session"""
    session_id: str
    harmful_percentages: List[int]
    focus_types: List[FocusType]
    intervention_types: List[InterventionType]
    base_output_dir: str
    max_parallelism: int

class DirectoryManager:
    """Manages directory structure for different stages and sessions"""
    
    def __init__(self, base_output_dir: str):
        self.base_output_dir = Path(base_output_dir).resolve()  # Make absolute
        self.ensure_base_directories()
    
    def ensure_base_directories(self):
        """Create base directory structure"""
        # Create base directory first
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create stage-specific directories
        for stage in ["train", "intervention"]:
            stage_dir = self.base_output_dir / stage
            stage_dir.mkdir(parents=True, exist_ok=True)
            
            for folder in [OUTPUT_DIR_NAME, LOGS_DIR_NAME, SHARED_DIR_NAME]:
                (stage_dir / folder).mkdir(parents=True, exist_ok=True)
    
    def get_stage_dirs(self, stage: StageType) -> Dict[str, str]:
        """Get absolute directory paths for a specific stage"""
        stage_base = self.base_output_dir / stage.value
        return {
            "output": str(stage_base / OUTPUT_DIR_NAME),
            "logs": str(stage_base / LOGS_DIR_NAME),
            "shared": str(stage_base / SHARED_DIR_NAME)
        }
    
    def get_docker_dirs(self, stage: StageType) -> Dict[str, str]:
        """Get Docker mount paths for a specific stage"""
        return {
            "output": f"/{stage.value}-{DOCKER_OUTPUT_DIR_PREFIX}",
            "logs": f"/{stage.value}-{DOCKER_LOGS_DIR_PREFIX}",
            "shared": f"/{stage.value}-{DOCKER_SHARED_DIR_PREFIX}"
        }
    
    def copy_puppet_profile(self, source_puppet_id: str, target_puppet_id: str):
        """Copy puppet profile from training to intervention stage"""
        source_profile = self.base_output_dir / "train" / OUTPUT_DIR_NAME / PROFILES_DIR_NAME / source_puppet_id
        target_profile = self.base_output_dir / "intervention" / OUTPUT_DIR_NAME / PROFILES_DIR_NAME / target_puppet_id
        
        logging.info(f"Copying profile from {source_profile} to {target_profile}")
        
        if source_profile.exists():
            target_profile.parent.mkdir(parents=True, exist_ok=True)

            import subprocess
            import os
            logging.info(f"Permission denied, using sudo to copy profile")
            sudo_password = os.getenv('SUDO_PASSWORD', '')
            try:
                if sudo_password:
                    # Use echo to pipe password to sudo -S
                    cmd = f'echo "{sudo_password}" | sudo -S cp -r "{source_profile}" "{target_profile}"'
                    subprocess.run(cmd, shell=True, check=True, capture_output=True)
                    
                    # Fix ownership
                    cmd = f'echo "{sudo_password}" | sudo -S chown -R {os.getenv("USER")}:{os.getenv("USER")} "{target_profile}"'
                    subprocess.run(cmd, shell=True, check=True, capture_output=True)
                else:
                    # Fallback to original method (will prompt for password)
                    subprocess.run([
                        "sudo", "cp", "-r", str(source_profile), str(target_profile)
                    ], check=True)
                    import getpass
                    current_user = getpass.getuser()
                    subprocess.run([
                        "sudo", "chown", "-R", f"{current_user}:{current_user}", str(target_profile)
                    ], check=True)
                
                logging.info(f"Copied profile from {source_profile} to {target_profile}")
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to copy profile even with sudo: {e}")
        else:
            logging.warning(f"Source profile not found: {source_profile}")
    
    def copy_puppet_state(self, source_puppet_id: str, target_puppet_id: str):
        """Copy puppet state from training to intervention stage"""
        source_state = self.base_output_dir / "train" / OUTPUT_DIR_NAME / PUPPETS_DIR_NAME / f"{source_puppet_id}.json"
        target_state = self.base_output_dir / "intervention" / OUTPUT_DIR_NAME / PUPPETS_DIR_NAME / f"{target_puppet_id}.json"
        
        if source_state.exists():
            target_state.parent.mkdir(parents=True, exist_ok=True)
            try:
                # Try normal copy first
                shutil.copy2(source_state, target_state)
                logging.info(f"Copied state from {source_state} to {target_state}")
            except PermissionError:
                # Use sudo if permission denied
                import subprocess
                import os
                logging.info(f"Permission denied, using sudo to copy state")
                sudo_password = os.getenv('SUDO_PASSWORD', '')
                try:
                    if sudo_password:
                        # Use echo to pipe password to sudo -S
                        cmd = f'echo "{sudo_password}" | sudo -S cp "{source_state}" "{target_state}"'
                        subprocess.run(cmd, shell=True, check=True, capture_output=True)
                        
                        # Fix ownership
                        cmd = f'echo "{sudo_password}" | sudo -S chown {os.getenv("USER")}:{os.getenv("USER")} "{target_state}"'
                        subprocess.run(cmd, shell=True, check=True, capture_output=True)
                    else:
                        # Fallback to original method
                        subprocess.run([
                            "sudo", "cp", str(source_state), str(target_state)
                        ], check=True)
                        import getpass
                        current_user = getpass.getuser()
                        subprocess.run([
                            "sudo", "chown", f"{current_user}:{current_user}", str(target_state)
                        ], check=True)
                    logging.info(f"Copied state from {source_state} to {target_state}")
                except subprocess.CalledProcessError as e:
                    logging.error(f"Failed to copy state even with sudo: {e}")
        else:
            logging.warning(f"Source state not found: {source_state}")
class PuppetFactory:
    """Factory for creating puppet configurations"""
    
    @staticmethod
    def create_training_puppets(session_config: SessionConfig, total_puppets: int) -> List[PuppetConfig]:
        """Create puppet configurations for training stage, cycling through harmful percentages"""
        puppets = []
        harmful_percentages = session_config.harmful_percentages
        for i in range(total_puppets):
            percentage = harmful_percentages[i % len(harmful_percentages)]  # Cycle through [0, 25, 50]
            puppet_id = f"harmful_{percentage}_{session_config.session_id}_{str(uuid4())[:8]}"
            puppets.append(PuppetConfig(
                puppet_id=puppet_id,
                stage=StageType.TRAIN,
                harmful_percentage=percentage,
                session_id=session_config.session_id
            ))
        logging.info(f"Created {len(puppets)} training puppets with percentages: {[p.harmful_percentage for p in puppets]}")
        return puppets
    
    @staticmethod
    def create_intervention_puppets(training_puppets: List[PuppetConfig], 
                                  session_config: SessionConfig) -> List[PuppetConfig]:
        """Create puppet configurations for intervention stage"""
        puppets = []
        for training_puppet in training_puppets:
            for focus in session_config.focus_types:
                for intervention in session_config.intervention_types:
                    puppet_id = f"{training_puppet.puppet_id}_{focus.value}_{intervention.value}"
                    puppets.append(PuppetConfig(
                        puppet_id=puppet_id,
                        stage=StageType.INTERVENTION,
                        harmful_percentage=training_puppet.harmful_percentage,
                        focus=focus,
                        intervention_type=intervention,
                        base_puppet_id=training_puppet.puppet_id,
                        session_id=session_config.session_id
                    ))
        return puppets

class VideoPoolManager:
    """Manages video pools for training"""
    
    def __init__(self, training_videos_path: str):
        self.training_videos_path = Path(training_videos_path)
        self.harmful_pool = None
        self.harmless_pool = None
        self.load_pools()
    
    def load_pools(self):
        """Load video pools from CSV files"""
        try:
            harmful_path = self.training_videos_path / "harmful.csv"
            harmless_path = self.training_videos_path / "non_harmful.csv"
            
            if not harmful_path.exists() or not harmless_path.exists():
                raise FileNotFoundError(f"Video pool files not found in {self.training_videos_path}")
            
            self.harmful_pool = pd.read_csv(harmful_path)
            self.harmless_pool = pd.read_csv(harmless_path)
            
            logging.info(f"Loaded {len(self.harmful_pool)} harmful and {len(self.harmless_pool)} harmless videos")
        except Exception as e:
            logging.error(f"Error loading video pools: {e}")
            raise
    
    def get_training_videos(self, harmful_percentage: int) -> List[str]:
        """Get training video list for specified harmful percentage"""
        harmful_count = int(NUM_TRAINING_VIDEOS * (harmful_percentage / 100))
        harmless_count = NUM_TRAINING_VIDEOS - harmful_count
        
        sampled_harmful = random.sample(
            self.harmful_pool["videoId"].tolist(), 
            min(harmful_count, len(self.harmful_pool))
        )
        sampled_harmless = random.sample(
            self.harmless_pool["videoId"].tolist(), 
            min(harmless_count, len(self.harmless_pool))
        )
        
        combined_videos = sampled_harmful + sampled_harmless
        random.shuffle(combined_videos)
        return combined_videos

class ContainerManager:
    """Manages Docker container lifecycle"""
    
    def __init__(self, max_containers: int, sleep_duration: int = 60):
        self.client = docker.from_env()
        self.max_containers = max_containers
        self.sleep_duration = sleep_duration
    
    def wait_for_capacity(self):
        """Wait until there's capacity for new containers"""
        while len(self.client.containers.list()) >= self.max_containers:
            logging.info(f"Max containers ({self.max_containers}) reached. Waiting...")
            sleep(self.sleep_duration)
    
    def get_mount_volumes(self, stage_dirs: Dict[str, str], docker_dirs: Dict[str, str]) -> Dict[str, Dict[str, str]]:
        """Get volume mounts for Docker container"""
        return {
            stage_dirs["output"]: {"bind": docker_dirs["output"]},
            stage_dirs["shared"]: {"bind": docker_dirs["shared"]},
            stage_dirs["logs"]: {"bind": docker_dirs["logs"]},
        }
    
    def run_puppet_container(self, puppet_config: PuppetConfig, puppet_args: Dict, 
                           volumes: Dict, network: str = "bridge") -> Optional[str]:
        """Run a single puppet container"""
        try:
            self.wait_for_capacity()
            
            command = ["python", "sockpuppet.py", json.dumps(puppet_args)]
            container = self.client.containers.run(
                IMAGE_NAME,
                command,
                volumes=volumes,
                shm_size="512M",
                remove=True,
                detach=True,
                network=network,
                extra_hosts={"host.docker.internal": "host-gateway"}
            )
            
            logging.info(f"Started container {container.id} for puppet {puppet_config.puppet_id}")
            sleep(3)  # Give some time for container to start

            return container.id
        except Exception as e:
            logging.error(f"Failed to start container for {puppet_config.puppet_id}: {e}")
            return None
    

class PuppetOrchestrator:
    """Main orchestrator for puppet experiments"""
    
    def __init__(self, args):
        self.args = args
        self.setup_logging()
        
        self.dir_manager = DirectoryManager(args.base_output_dir)
        self.video_manager = VideoPoolManager(args.training_videos)
        self.container_manager = ContainerManager(args.max_containers, args.sleep_duration)
        
        # Task queues and results
        self.training_queue = Queue()
        self.intervention_queue = Queue()
        self.completed_training = {}  # puppet_id -> PuppetConfig
        self.lock = threading.Lock()
    
    def setup_logging(self):
        """Setup logging configuration"""
        os.makedirs(self.args.base_output_dir, exist_ok=True)
        log_file = Path(self.args.base_output_dir) / "orchestrator.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def create_puppet_args(self, puppet_config: PuppetConfig, stage_dirs: Dict[str, str], 
                          docker_dirs: Dict[str, str]) -> Dict:
        """Create arguments dictionary for puppet"""
        base_args = {
            "puppetId": puppet_config.puppet_id,
            "duration": WATCH_DURATION,
            "description": f"Sock puppet {puppet_config.stage.value} - {puppet_config.harmful_percentage}% harmful",
            "harmful_percentage": puppet_config.harmful_percentage,
            "outputDir": docker_dirs["output"],
            "trainingN": NUM_TRAINING_VIDEOS,
            "steps": puppet_config.stage.value,
            "rounds": ROUNDS,
        }
        
        if puppet_config.stage == StageType.TRAIN:
            base_args["training"] = self.video_manager.get_training_videos(puppet_config.harmful_percentage)
        elif puppet_config.stage == StageType.INTERVENTION:
            base_args["focus"] = puppet_config.focus.value
            base_args["intervention_type"] = puppet_config.intervention_type.value
        
        return base_args
    
    def run_training_puppet(self, puppet_config: PuppetConfig) -> bool:
        """Run a single training puppet"""
        logging.info(f"Starting training puppet: {puppet_config.puppet_id}")
        
        stage_dirs = self.dir_manager.get_stage_dirs(StageType.TRAIN)
        docker_dirs = self.dir_manager.get_docker_dirs(StageType.TRAIN)
        
        # Create shared directory for puppet
        puppet_shared_dir = Path(stage_dirs["shared"]) / puppet_config.puppet_id
        puppet_shared_dir.mkdir(parents=True, exist_ok=True)
        
        puppet_args = self.create_puppet_args(puppet_config, stage_dirs, docker_dirs)
        volumes = self.container_manager.get_mount_volumes(stage_dirs, docker_dirs)
        
        logging.info(f"Training puppet volumes: {volumes}")
        
        if not self.args.simulate:
            container_id = self.container_manager.run_puppet_container(
                puppet_config, puppet_args, volumes, self.args.network
            )
            
            if container_id:
                # Wait for training to complete
                try:
                    container = self.container_manager.client.containers.get(container_id)
                    container.wait()
                    logging.info(f"Training completed for puppet {puppet_config.puppet_id}")
                    return True
                except Exception as e:
                    logging.error(f"Training failed for puppet {puppet_config.puppet_id}: {e}")
                    return False
        else:
            logging.info(f"Simulated training for puppet {puppet_config.puppet_id}")
            return True
        
        return False
    
    def run_intervention_puppet(self, puppet_config: PuppetConfig) -> bool:
        """Run a single intervention puppet"""
        logging.info(f"Starting intervention puppet: {puppet_config.puppet_id}")
        
        # Copy files from training stage
        self.dir_manager.copy_puppet_profile(puppet_config.base_puppet_id, puppet_config.puppet_id)
        self.dir_manager.copy_puppet_state(puppet_config.base_puppet_id, puppet_config.puppet_id)
        
        stage_dirs = self.dir_manager.get_stage_dirs(StageType.INTERVENTION)
        docker_dirs = self.dir_manager.get_docker_dirs(StageType.INTERVENTION)
        
        # Create shared directory for puppet
        puppet_shared_dir = Path(stage_dirs["shared"]) / puppet_config.puppet_id
        puppet_shared_dir.mkdir(parents=True, exist_ok=True)
        
        puppet_args = self.create_puppet_args(puppet_config, stage_dirs, docker_dirs)
        volumes = self.container_manager.get_mount_volumes(stage_dirs, docker_dirs)
        
        logging.info(f"Intervention puppet volumes: {volumes}")
        
        if not self.args.simulate:
            container_id = self.container_manager.run_puppet_container(
                puppet_config, puppet_args, volumes, self.args.network
            )
            return container_id is not None
        else:
            logging.info(f"Simulated intervention for puppet {puppet_config.puppet_id}")
            return True
    
    def run_session(self, session_config: SessionConfig):
        """Run a complete experimental session with prioritized interventions"""
        logging.info(f"Starting session {session_config.session_id}")
        total_training_puppets = session_config.max_parallelism  # Use max_containers
        training_puppets = PuppetFactory.create_training_puppets(session_config, total_puppets=total_training_puppets)
        intervention_puppets = PuppetFactory.create_intervention_puppets(training_puppets, session_config)
        logging.info(f"Session plan: {len(training_puppets)} training puppets, {len(intervention_puppets)} intervention puppets")
        
        intervention_map = {}
        for puppet in intervention_puppets:
            base_id = puppet.base_puppet_id
            if base_id not in intervention_map:
                intervention_map[base_id] = []
            intervention_map[base_id].append(puppet)
        
        pending_training_puppets = training_puppets.copy()
        completed_training_count = 0
        
        logging.info("=== Starting Pipeline Execution ===")
        with concurrent.futures.ThreadPoolExecutor(max_workers=session_config.max_parallelism) as executor:
            active_futures = {}
            
            # Step 1: Queue initial training puppets up to max_parallelism
            for i in range(min(session_config.max_parallelism, len(pending_training_puppets))):
                puppet = pending_training_puppets.pop(0)
                future = executor.submit(self.run_training_puppet, puppet)
                active_futures[future] = ('training', puppet)
            
            # Step 2: Process completions, prioritize interventions, refill with training
            while completed_training_count < total_training_puppets or active_futures:
                done_futures, _ = concurrent.futures.wait(active_futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
                
                for future in done_futures:
                    future_type, puppet = active_futures.pop(future)
                    
                    if future_type == 'training':
                        try:
                            success = future.result()
                            if success:
                                logging.info(f"Training completed: {puppet.puppet_id}")
                                completed_training_count += 1
                                # Queue interventions for this puppet
                                for intervention_puppet in intervention_map.get(puppet.puppet_id, []):
                                    int_future = executor.submit(self.run_intervention_puppet, intervention_puppet)
                                    active_futures[int_future] = ('intervention', intervention_puppet)
                            else:
                                logging.error(f"Training failed: {puppet.puppet_id}")
                        except Exception as e:
                            logging.error(f"Training error {puppet.puppet_id}: {e}")
                    
                    elif future_type == 'intervention':
                        try:
                            success = future.result()
                            if success:
                                logging.info(f"Intervention completed: {puppet.puppet_id}")
                            else:
                                logging.error(f"Intervention failed: {puppet.puppet_id}")
                        except Exception as e:
                            logging.error(f"Intervention error {puppet.puppet_id}: {e}")
                    
                    # Refill with new tasks, prioritizing interventions
                    while len(active_futures) < session_config.max_parallelism:
                        # Check for pending interventions in intervention_map
                        pending_interventions = []
                        for base_id, int_puppets in intervention_map.items():
                            if int_puppets:  # If there are still interventions for this training puppet
                                pending_interventions.extend(int_puppets)
                        
                        if pending_interventions:
                            # Queue an intervention puppet
                            int_puppet = intervention_map[pending_interventions[0].base_puppet_id].pop(0)
                            int_future = executor.submit(self.run_intervention_puppet, int_puppet)
                            active_futures[int_future] = ('intervention', int_puppet)
                        elif pending_training_puppets:
                            # Queue a new training puppet
                            puppet = pending_training_puppets.pop(0)
                            future = executor.submit(self.run_training_puppet, puppet)
                            active_futures[future] = ('training', puppet)
                        else:
                            break  # No more tasks to queue
        
        logging.info(f"Session {session_config.session_id} completed")
    
    def build_image(self):
        """Build Docker image"""
        logging.info("Building Docker image...")
        try:
            self.container_manager.client.images.build(
                path='./sockpuppet', 
                tag=IMAGE_NAME, 
                rm=True
            )
            logging.info("Docker image built successfully")
        except Exception as e:
            logging.error(f"Failed to build Docker image: {e}")
            raise

def parse_args():
    """Parse command line arguments"""
    parser = ArgumentParser(description="YouTube Sock Puppet Orchestrator")
    
    # Basic operations
    parser.add_argument("--build", action="store_true", help="Build docker image")
    parser.add_argument("--run", action="store_true", help="Run experimental sessions")
    parser.add_argument("--simulate", action="store_true", help="Simulate without running containers")
    
    # Session configuration
    parser.add_argument("--sessions", type=int, default=1, help="Number of complete sessions to run")
    parser.add_argument("--harmful-percentages", type=int, nargs="+", default=[0], 
                       help="Harmful content percentages for training")
    parser.add_argument("--focus-types", choices=["homepage", "upnext"], nargs="+", 
                       default=["homepage", "upnext"], help="Focus types for intervention")
    parser.add_argument("--intervention-types", choices=["none", "downrank", "replace"], nargs="+",
                       default=["none", "downrank", "replace"], help="Intervention types")
    #parser.add_argument("--focus-types", choices=["homepage", "upnext"], nargs="+", 
    #                   default=["homepage"], help="Focus types for intervention")
    #parser.add_argument("--intervention-types", choices=["none", "downrank", "replace"], nargs="+",
    #                   default=["none"], help="Intervention types")
    
    # Infrastructure
    parser.add_argument("--max-containers", type=int, default=10, help="Maximum concurrent containers")
    parser.add_argument("--sleep-duration", type=int, default=60, help="Sleep when max containers reached")
    parser.add_argument("--network", type=str, default="bridge", help="Docker network mode")
    
    # Paths
    parser.add_argument("--base-output-dir", default="experiments", help="Base output directory")
    parser.add_argument("--training-videos", default="data/training/", help="Training videos directory")
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_args()
    
    # Convert string lists to enums
    focus_types = [FocusType(f) for f in args.focus_types]
    intervention_types = [InterventionType(i) for i in args.intervention_types]
    
    orchestrator = PuppetOrchestrator(args)
    
    if args.build:
        orchestrator.build_image()
    
    if args.run:
        for session_num in range(args.sessions):
            session_id = f"session_{session_num}_{str(uuid4())[:8]}"
            
            session_config = SessionConfig(
                session_id=session_id,
                harmful_percentages=args.harmful_percentages,
                focus_types=focus_types,
                intervention_types=intervention_types,
                base_output_dir=args.base_output_dir,
                max_parallelism=args.max_containers
            )
            
            orchestrator.run_session(session_config)
    
    if not args.build and not args.run:
        logging.info("No operation specified. Use --build or --run.")

if __name__ == "__main__":
    main()
