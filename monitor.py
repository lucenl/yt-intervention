import os
import subprocess
from flask import Flask, request, jsonify
from multiprocessing import Process
import logging
import stat
import grp
import pwd

app = Flask(__name__)
SHARED_DIR = "./shared"
LOCAL_LOG_DIR = "./local_logs"
EXPERIMENT_DATA_DIR = "./experiment_data"
os.makedirs(LOCAL_LOG_DIR, exist_ok=True)
os.makedirs(SHARED_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOCAL_LOG_DIR, "monitor.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)
logger = logging.getLogger(__name__)

active_processes = {}

def run_preprocess(puppet_id, round_num, intervention_type, focus, training=None):
    """
    Run preprocess.py to generate recommendations and next video.
    """
    logger.info(f"run_preprocess() for round {round_num}")
    cmd = f"python preprocess.py {puppet_id} {round_num} {intervention_type} {focus}"
    if training:
        cmd += f" --training {','.join(training)}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Preprocessing failed for {puppet_id}, round {round_num}: {result.stderr}")
        return False
    logger.info(f"Preprocessing completed for {puppet_id}, round {round_num}")
    return True

@app.route('/start_preprocess', methods=['POST'])
def start_preprocess():
    global active_processes
    data = request.get_json()
    puppet_id = data.get('puppet_id')
    round_num = data.get('round_num')
    intervention_type = data.get('intervention_type', 'downrank')
    focus = data.get('focus', 'homepage')
    training = data.get('training')
    
    # if len(active_processes) >= max_processes:
    #     return jsonify({"status": "error", "message": "Max processes reached"}), 429

    logger.info(f"Received start_preprocess request for {puppet_id}, round {round_num}")
    process = Process(target=run_preprocess, args=(puppet_id, round_num, intervention_type, focus, training))
    logger.info(f"start_preprocess() for {puppet_id}, round {round_num}")
    process.start()
    key = f"{puppet_id}_{round_num}"
    active_processes[key] = process

    return jsonify({"status": "accepted", "process_id": key})

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    data = request.get_json()
    puppet_id = data.get('puppet_id')
    round_num = data.get('round_num')
    
    key = f"{puppet_id}_{round_num}"
    if key not in active_processes or active_processes[key].is_alive():
        return jsonify({"status": "pending", "message": "Preprocessing not yet complete"}), 202

    rec_file = os.path.join(SHARED_DIR, puppet_id, f"recommendations_{round_num}.txt")
    next_video_file = os.path.join(SHARED_DIR, puppet_id, f"next_video_{round_num}.txt")
    logger.info(f"get_recommendations() for {puppet_id}, round {round_num}")
    recommendations = []
    next_video = None

    if os.path.exists(rec_file):
        with open(rec_file, "r") as f:
            recommendations = [vid.strip() for vid in f.read().splitlines() if vid.strip()]
    else:
        logger.error(f"Recommendations file {rec_file} not found")
        return jsonify({"status": "error", "message": "Recommendations not found"}), 404

    if os.path.exists(next_video_file):
        with open(next_video_file, "r") as f:
            next_video = f.read().strip()
    else:
        logger.error(f"Next video file {next_video_file} not found due to permission error")
        return jsonify({"error": "Preprocessing failed, check logs"}), 500
    
    logger.info(f"Returning recommendations for {puppet_id}, round {round_num}: {recommendations}")
    return jsonify({"recommendations": recommendations, "next_video": next_video})

@app.route('/complete_round', methods=['POST'])
def complete_round():
    logger.info("complete_round(): Received complete_round request")
    data = request.get_json()
    puppet_id = data.get('puppet_id')
    round_num = data.get('round_num')
    key = f"{puppet_id}_{round_num}"
    if key in active_processes:
        if not active_processes[key].is_alive():
            del active_processes[key]
            logger.info(f"Completed round {round_num} for {puppet_id}")
            return jsonify({"status": "success"})
        else:
            return jsonify({"status": "pending", "message": "Process still running"}), 202
    
    logger.info("complete_round(): Complete complete_round request")
    return jsonify({"status": "error", "message": "Process not found"}), 404

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)