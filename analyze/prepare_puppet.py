"""
Prepare pre-trained puppets for intervention by converting metadata CSVs to experiment_log.json
and copying profiles and puppet state files to a new output directory.
"""

import os
import json
import pandas as pd
import subprocess
import logging
from pathlib import Path
import shutil

def setup_logging():
    """Set up logging for the preparation process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='prepare_puppet_data.log',
        filemode='a'
    )
    return logging.getLogger(__name__)

def load_puppet_actions(puppet_file):
    """
    Load the puppet file to extract recommendation actions.

    Args:
        puppet_file: Path to the puppet JSON file

    Returns:
        Tuple of (homepage_recommendations, upnext_recommendations)
    """
    logger = logging.getLogger(__name__)
    try:
        with open(puppet_file, 'r') as f:
            puppet_data = json.load(f)
        
        actions = puppet_data.get("actions", [])
        homepage_recommendations = []
        upnext_recommendations = []
        
        for action in reversed(actions):
            if action["action"] == "get_homepage_recommendations" and not homepage_recommendations:
                homepage_recommendations = action["params"] or []
            elif action["action"] == "get_upnext_recommendations" and not upnext_recommendations:
                upnext_recommendations = action["params"] or []
            if homepage_recommendations and upnext_recommendations:
                break
        
        homepage_recommendations = homepage_recommendations[:25]  # Match trimming in sockpuppet.py
        upnext_recommendations = upnext_recommendations[:12]
        
        logger.info(f"Loaded {len(homepage_recommendations)} homepage and {len(upnext_recommendations)} up-next recommendations from {puppet_file}")
        return homepage_recommendations, upnext_recommendations
    except Exception as e:
        logger.error(f"Error loading puppet file {puppet_file}: {e}")
        return [], []

def convert_csv_to_metadata_list(csv_path):
    """
    Convert a metadata CSV file to a list of metadata dictionaries.

    Args:
        csv_path: Path to the CSV file

    Returns:
        List of metadata dictionaries
    """
    logger = logging.getLogger(__name__)
    try:
        df = pd.read_csv(csv_path)
        metadata_list = df.to_dict('records')
        logger.info(f"Converted {csv_path} with {len(metadata_list)} entries")
        return metadata_list
    except Exception as e:
        logger.error(f"Error converting CSV {csv_path}: {e}")
        return []

def create_experiment_log(puppet_id, output_dir, homepage_recommendations, upnext_recommendations, homepage_metadata, upnext_metadata):
    """
    Create an experiment_log.json file with the converted data.

    Args:
        puppet_id: Puppet ID
        output_dir: Directory to save the experiment log (source or destination)
        homepage_recommendations: List of homepage video IDs
        upnext_recommendations: List of up-next video IDs
        homepage_metadata: List of homepage metadata dictionaries
        upnext_metadata: List of up-next metadata dictionaries

    Returns:
        Path to the created experiment_log.json file
    """
    logger = logging.getLogger(__name__)
    experiment_log = {"rounds": []}
    
    # Round 0 for homepage
    if homepage_recommendations:
        round_data = {
            "round_number": 0,
            "focus": "homepage",
            "recommendations": homepage_recommendations,
            "predictions": [None] * len(homepage_recommendations),
            "metadata": homepage_metadata
        }
        experiment_log["rounds"].append(round_data)
    
    # Round 0 for up-next
    if upnext_recommendations:
        round_data = {
            "round_number": 0,
            "focus": "up-next",
            "recommendations": upnext_recommendations,
            "predictions": [None] * len(upnext_recommendations),
            "metadata": upnext_metadata
        }
        experiment_log["rounds"].append(round_data)
    
    # Save to experiment_log.json
    metadata_dir = os.path.join(output_dir, puppet_id, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    log_file = os.path.join(metadata_dir, "experiment_log.json")
    
    with open(log_file, 'w') as f:
        json.dump(experiment_log, f, indent=4)
    
    logger.info(f"Saved experiment_log.json for puppet {puppet_id} with {len(experiment_log['rounds'])} rounds")
    return log_file

def convert_metadata(puppet_id, source_output_dir):
    """
    Convert metadata CSVs to experiment_log.json for a single puppet.

    Args:
        puppet_id: Puppet ID
        source_output_dir: Source output directory

    Returns:
        Tuple of (homepage_recommendations, upnext_recommendations, experiment_log_path)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Converting metadata for puppet {puppet_id}")
    
    # 1) Load the puppet's JSON (with the 'actions' list)
    puppet_file = os.path.join(source_output_dir, "puppets", puppet_id)
    if not os.path.isfile(puppet_file):
        logger.error(f"Error loading puppet file {puppet_file}: not a file")
        return [], [], None

    homepage_recommendations, upnext_recommendations = load_puppet_actions(puppet_file)

    # Load metadata CSVs
    metadata_dir = os.path.join(source_output_dir, puppet_id, "metadata")
    homepage_csv = os.path.join(metadata_dir, "metadata_homepage_round_0.csv")
    upnext_csv = os.path.join(metadata_dir, "metadata_upnext_round_0.csv")
    
    homepage_metadata = []
    upnext_metadata = []
    
    # Convert homepage metadata
    if os.path.exists(homepage_csv):
        homepage_metadata = convert_csv_to_metadata_list(homepage_csv)
        # Align metadata with recommendations
        metadata_dict = {item["video_id"]: item for item in homepage_metadata}
        homepage_metadata = [
            metadata_dict.get(vid, {
                "links": f"https://youtube.com/watch?v={vid}",
                "video_id": vid,
                "channel": "",
                "title": "",
                "description": "",
                "transcript": "",
                "date": ""
            })
            for vid in homepage_recommendations
        ]
    else:
        logger.warning(f"Homepage CSV {homepage_csv} not found")
        homepage_metadata = [
            {
                "links": f"https://youtube.com/watch?v={vid}",
                "video_id": vid,
                "channel": "",
                "title": "",
                "description": "",
                "transcript": "",
                "date": ""
            }
            for vid in homepage_recommendations
        ]
    
    # Convert up-next metadata
    if os.path.exists(upnext_csv):
        upnext_metadata = convert_csv_to_metadata_list(upnext_csv)
        # Align metadata with recommendations
        metadata_dict = {item["video_id"]: item for item in upnext_metadata}
        upnext_metadata = [
            metadata_dict.get(vid, {
                "links": f"https://youtube.com/watch?v={vid}",
                "video_id": vid,
                "channel": "",
                "title": "",
                "description": "",
                "transcript": "",
                "date": ""
            })
            for vid in upnext_recommendations
        ]
    else:
        logger.warning(f"Up-next CSV {upnext_csv} not found")
        upnext_metadata = [
            {
                "links": f"https://youtube.com/watch?v={vid}",
                "video_id": vid,
                "channel": "",
                "title": "",
                "description": "",
                "transcript": "",
                "date": ""
            }
            for vid in upnext_recommendations
        ]
    
    # Create experiment_log.json in the source directory
    experiment_log_path = create_experiment_log(
        puppet_id,
        source_output_dir,
        homepage_recommendations,
        upnext_recommendations,
        homepage_metadata,
        upnext_metadata
    )
    
    return homepage_recommendations, upnext_recommendations, experiment_log_path

def copy_experiment_log_only(puppet_id, source_output_dir, dest_output_dir, experiment_log_path):
    logger = logging.getLogger(__name__)
    logger.info(f"Copying experiment_log files for puppet {puppet_id}")
     # Source and destination paths
    source_puppet_dir = os.path.join(source_output_dir, puppet_id)
    dest_puppet_dir = os.path.join(dest_output_dir, puppet_id)
    
    
    if experiment_log_path and os.path.exists(experiment_log_path):
        os.makedirs(dest_puppet_dir, exist_ok=True)
        dest_log_file = os.path.join(dest_puppet_dir, "experiment_log.json")
        cmd = [
            "sudo", "rsync", "-a", "--progress",
            experiment_log_path,
            dest_log_file
        ]
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Copied experiment_log.json for puppet {puppet_id}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error copying experiment_log.json for {puppet_id}: {e}")
    else:
        logger.warning(f"experiment_log.json not found at {experiment_log_path}")

    
def copy_puppet_files(puppet_id, source_output_dir, dest_output_dir, experiment_log_path):
    """
    Copy profiles, puppet state, and experiment_log.json for a single puppet.

    Args:
        puppet_id: Puppet ID
        source_output_dir: Source output directory
        dest_output_dir: Destination output directory
        experiment_log_path: Path to the experiment_log.json file to copy
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Copying files for puppet {puppet_id}")
    
    # Source and destination paths
    source_puppet_dir = os.path.join(source_output_dir, puppet_id)
    dest_puppet_dir = os.path.join(dest_output_dir, puppet_id)
    
    # Check if source exists
    if not os.path.exists(source_puppet_dir):
        logger.warning(f"Source directory {source_puppet_dir} not found, skipping")
        return
    
    # Create destination directory structure
    os.makedirs(dest_puppet_dir, exist_ok=True)
    
    # Copy profiles directory
    source_profiles = os.path.join(source_output_dir, "profiles", puppet_id)
    dest_profiles_parent = os.path.join(dest_output_dir, "profiles")
    os.makedirs(dest_profiles_parent, exist_ok=True)
    
    if os.path.exists(source_profiles):
        cmd = [
            "sudo", "rsync", "-a", "--progress",
            source_profiles + "/",  # Source (trailing slash for contents)
            os.path.join(dest_profiles_parent, puppet_id)  # Destination
        ]
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Copied profiles for puppet {puppet_id}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error copying profiles for {puppet_id}: {e}")
    else:
        logger.warning(f"Profiles directory {source_profiles} not found")
    
    # Copy puppets directory (puppet state file)
    source_puppets = os.path.join(source_output_dir, "puppets", puppet_id)
    dest_puppets_parent = os.path.join(dest_output_dir, "puppets")
    os.makedirs(dest_puppets_parent, exist_ok=True)
    
    if os.path.exists(source_puppets):
        cmd = [
            "sudo", "rsync", "-a", "--progress",
            source_puppets,
            dest_puppets_parent + "/"
        ]
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Copied puppet state for puppet {puppet_id}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error copying puppet state for {puppet_id}: {e}")
    else:
        logger.warning(f"Puppet state file {source_puppets} not found")
    
    # Copy experiment_log.json
    if experiment_log_path and os.path.exists(experiment_log_path):
        os.makedirs(dest_puppet_dir, exist_ok=True)
        dest_log_file = os.path.join(dest_puppet_dir, "experiment_log.json")
        cmd = [
            "sudo", "rsync", "-a", "--progress",
            experiment_log_path,
            dest_log_file
        ]
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Copied experiment_log.json for puppet {puppet_id}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error copying experiment_log.json for {puppet_id}: {e}")
    else:
        logger.warning(f"experiment_log.json not found at {experiment_log_path}")

def copy_metadata_csvs(puppet_id, source_root, dest_root):
    """
    Copy only the two “metadata_*.csv” files from source to dest.
    """
    logger = logging.getLogger(__name__)
    src_meta = Path(source_root) / puppet_id / "metadata"
    dst_meta = Path(dest_root) / puppet_id / "metadata"
    if not src_meta.exists():
        logger.warning(f"No metadata folder in source for {puppet_id}, skipping CSV copy")
        return

    dst_meta.mkdir(parents=True, exist_ok=True)
    for name in ("metadata_homepage_round_0.csv", "metadata_upnext_round_0.csv"):
        src = src_meta / name
        dst = dst_meta / name
        if src.exists():
            try:
                shutil.copy2(src, dst)
                logger.info(f"Copied {name} for {puppet_id}")
            except Exception as e:
                logger.error(f"Failed to copy {src} → {dst}: {e}")
        else:
            logger.warning(f"{name} not found for {puppet_id}; skipping")
            

def prepare_puppet(puppet_id, source_output_dir, dest_output_dir):
    """
    Prepare a single puppet by converting metadata and copying files.

    Args:
        puppet_id: Puppet ID
        source_output_dir: Source output directory
        dest_output_dir: Destination output directory
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Preparing puppet {puppet_id}")
    
    copy_metadata_csvs(puppet_id, source_output_dir, dest_output_dir)
    
    # Step 1: Convert metadata to experiment_log.json
    homepage_recommendations, upnext_recommendations, experiment_log_path = convert_metadata(puppet_id, dest_output_dir)
    # copy_experiment_log_only(puppet_id, source_output_dir, dest_output_dir, experiment_log_path)
    
    # Step 2: Copy profiles, puppet state, and experiment_log.json
    # copy_puppet_files(puppet_id, source_output_dir, dest_output_dir, experiment_log_path)

def main():
    """Main function to prepare specified pre-trained puppets."""
    logger = setup_logging()
    
    # Define paths
    source_output_dir = "/media/data/lucen/codebase/yt-sock-puppet/output_final"
    dest_output_dir = "/media/data/lucen/codebase/yt-sock-puppet/output_final_copy"
    
    # List of puppet IDs to prepare (adjust as needed)
    puppet_ids = [
        "harmful_50,534096cb",
        "harmful_50,736f881f",
        "harmful_50,50c164a0",
        "harmful_50,ca41dcc9",
        "harmful_50,ce900552",
        "harmful_50,71754cb0",
        "harmful_50,691a1b6d",
        "harmful_50,663e1589",
        "harmful_50,7a4df0f1",
        "harmful_50,593fbd05"
        # Add more puppet IDs as needed
    ]
    
    # Prepare each puppet
    for puppet_id in puppet_ids:
        prepare_puppet(puppet_id, source_output_dir, dest_output_dir)

if __name__ == "__main__":
    main()