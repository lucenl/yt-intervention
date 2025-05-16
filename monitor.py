import os
import time
import subprocess
from multiprocessing import Process
import logging

SHARED_DIR = "./shared"
LOG_DIR = "./local_logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "monitor.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

def run_preprocess(puppet_id, round_num):
    cmd = f"python preprocess.py {puppet_id} {round_num}  {focus}"
    subprocess.run(cmd, shell=True)

def monitor():
    processed = {}
    active_processes = {}
    max_processes = 4

    while True:
        for key in list(active_processes.keys()):
            process = active_processes[key]
            if not process.is_alive():
                process.join()
                del active_processes[key]

        for puppet_dir in os.listdir(SHARED_DIR):
            puppet_shared_dir = os.path.join(SHARED_DIR, puppet_dir)
            if not os.path.isdir(puppet_shared_dir):
                continue
            if puppet_dir not in processed:
                processed[puppet_dir] = set()
            for round_file in os.listdir(puppet_shared_dir):
                if round_file.startswith("ready_") and round_file not in processed[puppet_dir]:
                    if len(active_processes) >= max_processes:
                        break
                    round_num = int(round_file.split("_")[1].split(".")[0])
                    key = f"{puppet_dir}_{round_num}"
                    logging.info(f"Starting preprocessing for {puppet_dir}, round {round_num}")
                    process = Process(target=run_preprocess, args=(puppet_dir, round_num))
                    process.start()
                    active_processes[key] = process
                    processed[puppet_dir].add(round_file)

        time.sleep(5)

if __name__ == "__main__":
    import sys
    intervention_type = sys.argv[1] if len(sys.argv) > 1 else "downrank"
    selection_type = sys.argv[2] if len(sys.argv) > 2 else "decay_weighted_random"
    focus = sys.argv[3] if len(sys.argv) > 3 else "homepage"
    monitor()