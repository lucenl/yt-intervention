import os
import subprocess
from flask import Flask, request, jsonify
from multiprocessing import Process
import logging
from pathlib import Path
import json

app = Flask(__name__)

# Configuration that can be set via environment variables
BASE_OUTPUT_DIR = os.getenv('BASE_OUTPUT_DIR', 'experiments')
MONITOR_LOG_DIR = os.getenv('MONITOR_LOG_DIR', 'monitor-logs')

# Directory name constants (must match orchestrator and sockpuppet)
OUTPUT_DIR_NAME = "debug-output"
LOGS_DIR_NAME = "logs"
SHARED_DIR_NAME = "debug-shared"
PROFILES_DIR_NAME = "profiles"
PUPPETS_DIR_NAME = "puppets"

# Ensure monitor log directory exists
os.makedirs(MONITOR_LOG_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(
    filename=os.path.join(MONITOR_LOG_DIR, "monitor.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)
logger = logging.getLogger(__name__)

# Also log to console for debugging
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

active_processes = {}

class DirectoryResolver:
    """Resolves directory paths for different stages and puppets"""
    
    def __init__(self, base_output_dir: str):
        self.base_output_dir = Path(base_output_dir).resolve()
    
    def get_stage_from_puppet_id(self, puppet_id: str) -> str:
        """Determine stage from puppet ID"""
        # Intervention puppets have focus and intervention type suffixes
        intervention_suffixes = [
            "_homepage_none", "_homepage_downrank", "_homepage_replace",
            "_upnext_none", "_upnext_downrank", "_upnext_replace"
        ]
        
        for suffix in intervention_suffixes:
            if puppet_id.endswith(suffix):
                return "intervention"
        
        return "train"
    
    def get_stage_dirs(self, stage: str) -> dict:
        """Get directory paths for a specific stage"""
        stage_base = self.base_output_dir / stage
        return {
            "output": str(stage_base / OUTPUT_DIR_NAME),
            "logs": str(stage_base / LOGS_DIR_NAME),
            "shared": str(stage_base / SHARED_DIR_NAME)
        }
    
    def get_puppet_shared_dir(self, puppet_id: str) -> str:
        """Get shared directory for a specific puppet"""
        stage = self.get_stage_from_puppet_id(puppet_id)
        stage_dirs = self.get_stage_dirs(stage)
        return os.path.join(stage_dirs["shared"], puppet_id)
    
    def ensure_puppet_shared_dir(self, puppet_id: str) -> str:
        """Ensure puppet shared directory exists and return path"""
        puppet_shared_dir = self.get_puppet_shared_dir(puppet_id)
        os.makedirs(puppet_shared_dir, exist_ok=True)
        return puppet_shared_dir

# Initialize directory resolver
dir_resolver = DirectoryResolver(BASE_OUTPUT_DIR)

def run_preprocess(puppet_id, round_num, intervention_type, focus, training=None):
    """
    Run preprocess.py to generate recommendations and next video.
    """
    logger.info(f"Starting preprocessing for puppet {puppet_id}, round {round_num}")
    
    # Build command
    cmd_parts = ["python", "preprocess.py", puppet_id, str(round_num), intervention_type, focus]
    if training:
        cmd_parts.extend(["--training", ','.join(training)])
    
    # Set environment variables for preprocess.py to find directories
    env = os.environ.copy()
    env['BASE_OUTPUT_DIR'] = BASE_OUTPUT_DIR
    
    try:
        result = subprocess.run(
            cmd_parts, 
            capture_output=True, 
            text=True, 
            env=env
        )
        
        if result.returncode != 0:
            logger.error(f"Preprocessing failed for {puppet_id}, round {round_num}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return False
        
        logger.info(f"Preprocessing completed successfully for {puppet_id}, round {round_num}")
        logger.debug(f"Preprocessing output: {result.stdout}")
        return True
        
    except subprocess.TimeoutExpired:
        logger.error(f"Preprocessing timed out for {puppet_id}, round {round_num}")
        return False
    except Exception as e:
        logger.error(f"Preprocessing error for {puppet_id}, round {round_num}: {e}")
        return False

@app.route('/start_preprocess', methods=['POST'])
def start_preprocess():
    """Start preprocessing for a puppet round"""
    global active_processes
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        puppet_id = data.get('puppet_id')
        round_num = data.get('round_num')
        intervention_type = data.get('intervention_type', 'downrank')
        focus = data.get('focus', 'homepage')
        training = data.get('training')
        
        if not puppet_id or round_num is None:
            return jsonify({"error": "puppet_id and round_num are required"}), 400
        
        logger.info(f"Received start_preprocess request for {puppet_id}, round {round_num}")
        logger.info(f"Intervention: {intervention_type}, Focus: {focus}")
        
        # Ensure puppet shared directory exists
        puppet_shared_dir = dir_resolver.ensure_puppet_shared_dir(puppet_id)
        logger.info(f"Puppet shared directory: {puppet_shared_dir}")
        
        # Start preprocessing in background
        process = Process(
            target=run_preprocess, 
            args=(puppet_id, round_num, intervention_type, focus, training)
        )
        process.start()
        
        key = f"{puppet_id}_{round_num}"
        active_processes[key] = process
        
        logger.info(f"Started preprocessing process for {key}")
        return jsonify({"status": "accepted", "process_id": key})
        
    except Exception as e:
        logger.error(f"Error in start_preprocess: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    """Get recommendations and next video for a puppet round"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        puppet_id = data.get('puppet_id')
        round_num = data.get('round_num')
        
        if not puppet_id or round_num is None:
            return jsonify({"error": "puppet_id and round_num are required"}), 400
        
        key = f"{puppet_id}_{round_num}"
        logger.info(f"Checking recommendations for {key}")
        
        # Check if process is still running
        if key in active_processes and active_processes[key].is_alive():
            logger.info(f"Preprocessing still running for {key}")
            return jsonify({"status": "pending", "message": "Preprocessing not yet complete"}), 202
        
        # Check if process failed
        if key in active_processes and active_processes[key].exitcode != 0:
            logger.error(f"Preprocessing failed for {key}")
            del active_processes[key]
            return jsonify({"error": "Preprocessing failed, check logs"}), 500
        
        # Get puppet shared directory
        puppet_shared_dir = dir_resolver.get_puppet_shared_dir(puppet_id)
        logger.info(f"Looking for files in: {puppet_shared_dir}")
        
        rec_file = os.path.join(puppet_shared_dir, f"recommendations_{round_num}.txt")
        next_video_file = os.path.join(puppet_shared_dir, f"next_video_{round_num}.txt")
        
        recommendations = []
        next_video = None
        
        # Read recommendations
        if os.path.exists(rec_file):
            try:
                with open(rec_file, "r") as f:
                    recommendations = [vid.strip() for vid in f.read().splitlines() if vid.strip()]
                logger.info(f"Found {len(recommendations)} recommendations in {rec_file}")
            except Exception as e:
                logger.error(f"Error reading recommendations file {rec_file}: {e}")
                return jsonify({"error": "Error reading recommendations"}), 500
        else:
            logger.error(f"Recommendations file not found: {rec_file}")
            return jsonify({"error": "Recommendations not found"}), 404
        
        # Read next video
        if os.path.exists(next_video_file):
            try:
                with open(next_video_file, "r") as f:
                    next_video = f.read().strip()
                logger.info(f"Found next video: {next_video}")
            except Exception as e:
                logger.error(f"Error reading next video file {next_video_file}: {e}")
                return jsonify({"error": "Error reading next video"}), 500
        else:
            logger.error(f"Next video file not found: {next_video_file}")
            return jsonify({"error": "Next video not found"}), 500
        
        # Clean up completed process
        if key in active_processes:
            del active_processes[key]
        
        logger.info(f"Returning recommendations and next video for {key}")
        return jsonify({
            "recommendations": recommendations, 
            "next_video": next_video,
            "status": "complete"
        })
        
    except Exception as e:
        logger.error(f"Error in get_recommendations: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/complete_round', methods=['POST'])
def complete_round():
    """Mark a round as completed"""
    try:
        logger.info("Received complete_round request")
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        puppet_id = data.get('puppet_id')
        round_num = data.get('round_num')
        
        if not puppet_id or round_num is None:
            return jsonify({"error": "puppet_id and round_num are reaquired"}), 400
        
        key = f"{puppet_id}_{round_num}"
        
        if key in active_processes:
            if not active_processes[key].is_alive():
                del active_processes[key]
                logger.info(f"Completed round {round_num} for {puppet_id}")
                return jsonify({"status": "success"})
            else:
                logger.warning(f"Process still running for {key}")
                return jsonify({"status": "pending", "message": "Process still running"}), 202
        
        logger.info(f"Round {round_num} for {puppet_id} completed (process not found)")
        return jsonify({"status": "success"})
        
    except Exception as e:
        logger.error(f"Error in complete_round: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/initialize_round', methods=['POST'])
def initialize_round():
    """Initialize a round with recommendations (for testing/debugging)"""
    try:
        logger.info("Received initialize_round request")
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        puppet_id = data.get('puppet_id')
        round_num = data.get('round_num')
        recs = data.get('recommendations')
        
        if not puppet_id or round_num is None or not recs:
            return jsonify({"error": "puppet_id, round_num, and recommendations are required"}), 400
        
        # Ensure puppet shared directory exists
        puppet_shared_dir = dir_resolver.ensure_puppet_shared_dir(puppet_id)
        
        # Store recommendations
        rec_file = os.path.join(puppet_shared_dir, f"recommendations_{round_num}.txt")
        with open(rec_file, "w") as f:
            f.write("\n".join(recs))
        
        logger.info(f"Initialized round {round_num} for {puppet_id} with {len(recs)} recommendations")
        return jsonify({"status": "success"})
        
    except Exception as e:
        logger.error(f"Error in initialize_round: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "active_processes": len(active_processes),
        "base_output_dir": BASE_OUTPUT_DIR
    })

@app.route('/status', methods=['GET'])
def status():
    """Get status of active processes"""
    process_status = {}
    for key, process in active_processes.items():
        process_status[key] = {
            "alive": process.is_alive(),
            "pid": process.pid,
            "exitcode": process.exitcode
        }
    
    return jsonify({
        "active_processes": process_status,
        "base_output_dir": BASE_OUTPUT_DIR
    })

if __name__ == "__main__":
    logger.info(f"Starting monitor server with base output directory: {BASE_OUTPUT_DIR}")
    logger.info(f"Monitor logs directory: {MONITOR_LOG_DIR}")
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=5005, debug=False)
