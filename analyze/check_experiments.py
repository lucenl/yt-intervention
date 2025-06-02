#!/usr/bin/env python3
"""
Lightning-fast experiment processor with incremental processing and status tracking.
Avoids repeated work by tracking what's already been processed.
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
from collections import defaultdict
import hashlib
from datetime import datetime


class ProcessingTracker:
    """
    Tracks processing status to avoid repeated work.
    Uses a simple JSON file to store processing state.
    """
    
    def __init__(self, base_dir, cache_file=".processing_cache.json"):
        self.base_dir = Path(base_dir)
        self.cache_file = Path(__file__).parent / cache_file
        self.cache = self._load_cache()
    
    def _load_cache(self):
        """Load existing cache or create new one."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                print(f"Loaded processing cache with {len(cache.get('processed_puppets', {}))} tracked puppets")
                return cache
            except Exception as e:
                print(f"Warning: Could not load cache file: {e}")
        
        return {
            "processed_puppets": {},  # puppet_id -> {"status": "success|failed", "last_check": timestamp, "combined": bool}
            "last_stats_run": None,
            "expected_rounds": None,
            "version": "1.0"
        }
    
    def save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
    
    def get_unprocessed_puppets(self, all_puppet_dirs, expected_rounds):
        """Get list of puppet directories that haven't been processed yet."""
        processed = set(self.cache["processed_puppets"].keys())
        
        # If expected_rounds changed, reprocess everything
        if self.cache.get("expected_rounds") != expected_rounds:
            print(f"Expected rounds changed from {self.cache.get('expected_rounds')} to {expected_rounds}, reprocessing all")
            self.cache["expected_rounds"] = expected_rounds
            return all_puppet_dirs
        
        unprocessed = [puppet_id for puppet_id in all_puppet_dirs if puppet_id not in processed]
        
        if unprocessed:
            print(f"Found {len(unprocessed)} new puppets to check (skipping {len(processed)} already processed)")
        else:
            print(f"No new puppets found ({len(processed)} already processed)")
        
        return unprocessed
    
    def update_puppet_status(self, puppet_id, is_successful, combined=False):
        """Update status for a puppet."""
        self.cache["processed_puppets"][puppet_id] = {
            "status": "success" if is_successful else "failed",
            "last_check": datetime.now().isoformat(),
            "combined": combined
        }
    
    def get_successful_puppets(self):
        """Get list of successful puppet IDs."""
        return [
            puppet_id for puppet_id, info in self.cache["processed_puppets"].items()
            if info["status"] == "success"
        ]
    
    def get_failed_puppets(self):
        """Get list of failed puppet IDs."""
        return [
            puppet_id for puppet_id, info in self.cache["processed_puppets"].items()
            if info["status"] == "failed"
        ]
    
    def get_uncombined_successful_puppets(self):
        """Get successful puppets that haven't been combined yet."""
        return [
            puppet_id for puppet_id, info in self.cache["processed_puppets"].items()
            if info["status"] == "success" and not info.get("combined", False)
        ]
    
    def mark_combined(self, puppet_id):
        """Mark a puppet as having been combined."""
        if puppet_id in self.cache["processed_puppets"]:
            self.cache["processed_puppets"][puppet_id]["combined"] = True
    
    def update_stats_timestamp(self):
        """Update the last stats run timestamp."""
        self.cache["last_stats_run"] = datetime.now().isoformat()


def load_existing_lists(success_file, failed_file):
    """Load existing success/failed lists if they exist."""
    successful_set = set()
    failed_set = set()
    
    if Path(success_file).exists():
        with open(success_file, 'r', encoding='utf-8') as f:
            successful_set = {line.strip() for line in f if line.strip()}
        print(f"Loaded {len(successful_set)} successful puppets from existing file")
    
    if Path(failed_file).exists():
        with open(failed_file, 'r', encoding='utf-8') as f:
            failed_set = {line.strip() for line in f if line.strip()}
        print(f"Loaded {len(failed_set)} failed puppets from existing file")
    
    return successful_set, failed_set

def incremental_stats_analysis(base_dir, expected_rounds, output_prefix="experiment_stats", use_shell=False):
    """
    Incremental stats analysis - only check new puppets.
    """
    print("Running incremental stats analysis...")
    start_time = time.time()
    
    tracker = ProcessingTracker(base_dir)
    success_file = f"{output_prefix}_successful.txt"
    failed_file = f"{output_prefix}_failed.txt"
    
    try:
        # Get all puppet directories
        all_puppets = [d.name for d in Path(base_dir).iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        # Get puppets that haven't been processed yet
        unprocessed_puppets = tracker.get_unprocessed_puppets(all_puppets, expected_rounds)
        
        if not unprocessed_puppets:
            # No new puppets, just report existing stats
            successful_puppets = tracker.get_successful_puppets()
            failed_puppets = tracker.get_failed_puppets()
            
            # **MODIFIED**: Update text files as snapshots even when no new processing
            success_file = f"{output_prefix}_successful.txt"
            failed_file = f"{output_prefix}_failed.txt"
            
            with open(success_file, 'w', encoding='utf-8') as f:
                for puppet_id in sorted(successful_puppets):
                    f.write(f"{puppet_id}\n")
            
            with open(failed_file, 'w', encoding='utf-8') as f:
                for puppet_id in sorted(failed_puppets):
                    f.write(f"{puppet_id}\n")
            
            print(f"No new puppets to process. Current stats:")
            print(f"Total puppets: {len(all_puppets)}")
            print(f"Successful: {len(successful_puppets)}")
            print(f"Failed: {len(failed_puppets)}")
            if len(all_puppets) > 0:
                print(f"Success rate: {len(successful_puppets)/len(all_puppets)*100:.1f}%")
            
            print(f"Text files updated as snapshots:")
            print(f"  Successful puppets: {success_file}")
            print(f"  Failed puppets: {failed_file}")
            
            return all_puppets, all_successful, all_failed
        
        # Process only new puppets
        new_successful = []
        new_failed = []
        
        if use_shell:
            # Use find command for batch checking - much faster for many files
            if unprocessed_puppets:
                try:
                    # Build find command to check all final round files at once
                    puppet_dirs = [str(Path(base_dir) / puppet_id) for puppet_id in unprocessed_puppets]
                    
                    # Use find to check all directories at once
                    find_cmd = ["find"] + puppet_dirs + ["-maxdepth", "1", "-name", f"round_{expected_rounds}.json", "-type", "f"]
                    result = subprocess.run(find_cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        existing_files = set(result.stdout.strip().split('\n')) if result.stdout.strip() else set()
                    else:
                        existing_files = set()
                    
                    # Categorize puppets based on shell results
                    for puppet_id in unprocessed_puppets:
                        expected_file = str(Path(base_dir) / puppet_id / f"round_{expected_rounds}.json")
                        if expected_file in existing_files:
                            new_successful.append(puppet_id)
                            tracker.update_puppet_status(puppet_id, True)
                        else:
                            new_failed.append(puppet_id)
                            tracker.update_puppet_status(puppet_id, False)
                            
                except subprocess.CalledProcessError as e:
                    print(f"Shell command failed: {e}, falling back to Python method")
                    # Fall back to Python method if shell fails
                    use_shell = False
        
        # Get all successful and failed puppets (old + new)
        all_successful = tracker.get_successful_puppets()
        all_failed = tracker.get_failed_puppets()
        
        # Update the text files with complete lists
        with open(success_file, 'w', encoding='utf-8') as f:
            for puppet_id in sorted(all_successful):
                f.write(f"{puppet_id}\n")
        
        with open(failed_file, 'w', encoding='utf-8') as f:
            for puppet_id in sorted(all_failed):
                f.write(f"{puppet_id}\n")
        
        
        # Save cache
        tracker.update_stats_timestamp()
        tracker.save_cache()
        
        end_time = time.time()
        
        print(f"INCREMENTAL STATS (completed in {end_time - start_time:.2f} seconds):")
        print(f"New puppets processed: {len(unprocessed_puppets)}")
        print(f"  New successful: {len(new_successful)}")
        print(f"  New failed: {len(new_failed)}")
        print(f"\nTotal stats:")
        print(f"Total puppets: {len(all_puppets)}")
        print(f"Successful: {len(all_successful)}")
        print(f"Failed: {len(all_failed)}")
        if len(all_puppets) > 0:
            print(f"Success rate: {len(all_successful)/len(all_puppets)*100:.1f}%")
        
        print(f"\nText files updated as snapshots:")
        print(f"  Successful puppets: {success_file}")
        print(f"  Failed puppets: {failed_file}")
        
        return all_puppets, all_successful, all_failed
        
    except Exception as e:
        print(f"Error in incremental stats: {e}")
        return None

def collect_puppet_args(base_dir, output_dir):
    """
    Collect metadata from output folder and add to cache.
    Only processes puppets that don't have metadata yet.
    """
    tracker = ProcessingTracker(base_dir)
    
    print("Collecting puppet arguments...")
    start_time = time.time()
    
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"Output directory {output_dir} not found, skipping arguments collection")
        return
    
    processed_count = 0
    
    # Only check puppets we know are successful and don't have metadata
    successful_puppets = tracker.get_successful_puppets()
    failed_puppets = tracker.get_failed_puppets()
    
    # Map source dir to source name
    SOURCE_MAP = {
        "/media/data/lucen/codebase/yt-sock-puppet/output/puppets": "Lucen",
        "/media/data/lucen/codebase/yt-sock-puppet/output/synced-puppets/puppets": "Haroon",
        "/media/data/lucen/codebase/yt-sock-puppet/output/wv/puppets": "WV",
    }
  
    puppet_id_to_source = {}
    for src_dir, source_name in SOURCE_MAP.items():
        path = Path(src_dir).resolve()
        num_puppets = 0
        for file in path.glob("*.json"):
            puppet_id = file.stem
            puppet_id_to_source[puppet_id] = source_name
            num_puppets += 1

        print(f"Found {num_puppets} puppets in output directory: {src_dir}")
        
    for puppet_id in successful_puppets + failed_puppets:

        source_name = puppet_id_to_source.get(puppet_id)
        if source_name:
            tracker.cache["processed_puppets"].setdefault(puppet_id, {})["source"] = source_name


        # Skip if already has metadata
        puppet_info = tracker.cache["processed_puppets"].get(puppet_id, {})
        if puppet_info.get("arguments"):
            continue
            
        # Look for puppet's output file
        puppet_output_file = output_path / f"{puppet_id}.json"
        if puppet_output_file.exists():
            try:
                with open(puppet_output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Extract metadata from args
                args = data.get("args", {})
                arguments = {
                    "harmful_percentage": args.get("harmful_percentage"),
                    "duration": args.get("duration"),
                    "trainingN": args.get("trainingN"),
                    "steps": args.get("steps"),
                    "rounds": args.get("rounds"),
                    "focus": args.get("focus"),
                    "intervention_type": args.get("intervention_type")
                }
                
                # Add metadata to cache
                tracker.cache["processed_puppets"][puppet_id]["arguments"] = arguments
                processed_count += 1
                
            except Exception as e:
                print(f"Error reading arguments for {puppet_id}: {e}")
    
    tracker.save_cache()
    
    end_time = time.time()
    print(f"Collected metadata for {processed_count} puppets in {end_time - start_time:.2f} seconds")
    
def summarize_puppet_distribution(base_dir):
    tracker = ProcessingTracker(base_dir)
    data = []

    for puppet_id, info in tracker.cache["processed_puppets"].items():
        if info.get("status") != "success":
            continue
        args = info.get("arguments", {})
        hp = args.get("harmful_percentage")
        intervention = args.get("intervention_type")
        focus = args.get("focus")

        if hp is not None and intervention and focus:
            data.append((hp, focus, intervention))

    if not data:
        print("No valid puppet metadata available for summary.")
        return

    # Aggregate counts
    from collections import defaultdict
    summary = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for hp, focus, intervention in data:
        summary[hp][focus][intervention] += 1

    # Print summary matrix
    all_foci = sorted({f for _, f, _ in data})
    all_interventions = sorted({i for _, _, i in data})

    print("\n=== Puppet Distribution Summary ===")
    for hp in sorted(summary.keys()):
        print(f"\nHarmful Percentage = {hp}%")
        header = " " * 15 + " | " + " | ".join(f"{f:>10}" for f in all_foci)
        print(header)
        print("-" * len(header))
        for intervention in all_interventions:
            row = f"{intervention:>15} | "
            for focus in all_foci:
                count = summary[hp][focus].get(intervention, 0)
                row += f"{count:>10} | "
            print(row)


def incremental_combine_files(base_dir, expected_rounds, output_dir, success_list_file=None, max_workers=None):
    """
    Simple, efficient incremental file combination.
    Only combines puppets that don't already have combined files.
    """
    print("Running incremental file combination...")
    start_time = time.time()
    
    base_path = Path(base_dir)
    tracker = ProcessingTracker(base_dir)  # **MODIFIED**: Always create tracker
    
    # Load puppet cache
    if success_list_file:
        # Load custom/filtered list for subset processing
        with open(success_list_file, 'r', encoding='utf-8') as f:
            custom_puppet_ids = [line.strip() for line in f if line.strip()]
        # Get successful puppets from cache and filter by custom list
        cached_successful = set(tracker.get_successful_puppets())
        puppet_ids = [pid for pid in custom_puppet_ids if pid in cached_successful]
        print(f"Using custom list: {len(custom_puppet_ids)} puppets, {len(puppet_ids)} are successful in cache")
    else:
        # **MODIFIED**: Default behavior - use cache directly
        puppet_ids = tracker.get_successful_puppets()
        print(f"Using {len(puppet_ids)} successful puppets from cache")
    
    if not puppet_ids:
        print("No successful puppets found to combine.")
        return 0, 0, 0, []
    
    # Set up output directory
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Check which puppets need combining (incremental check)
    puppets_to_combine = []
    already_combined = 0
    
    for puppet_id in puppet_ids:
        # Check cache first - much faster than filesystem check
        if tracker.cache["processed_puppets"].get(puppet_id, {}).get("combined", False):
            already_combined += 1
        else:
            # Double-check filesystem only if cache says not combined
            combined_file = output_path / f"{puppet_id}.json"
           
            if combined_file.exists():
                already_combined += 1
                tracker.mark_combined(puppet_id)  # Update cache to sync with reality
            else:
                puppets_to_combine.append(puppet_id)
    
    print(f"Found {already_combined} already combined puppets (skipping)")
    print(f"Need to combine {len(puppets_to_combine)} puppets")
    
    if not puppets_to_combine:
        print("All successful puppets already have combined files!")
        tracker.save_cache()
        return len(puppet_ids), already_combined, 0, []
    
    # Simple combination loop
    successful_writes = 0
    
    for puppet_id in puppets_to_combine:
        try:
            puppet_data = {
                "puppet_id": puppet_id,
                "total_rounds": expected_rounds,
                "rounds": {}
            }
            
            puppet_folder = base_path / puppet_id
            missing_files = 0
            
            valid = True
            # Read all round files
            for round_num in range(1, expected_rounds + 1):
                round_file = puppet_folder / f"round_{round_num}.json"
                if round_file.exists():
                    try:
                        with open(round_file, 'r', encoding='utf-8') as f:
                            round_data = json.load(f)
                            puppet_data["rounds"][f"round_{round_num}"] = round_data
                    except json.JSONDecodeError as e:
                        print(f"JSONDecodeError: Skipping {round_file}: {e}")
                        missing_files += 1
                        valid = False
                        break
                
                else:
                    print(f"Missing round file: {round_file}, skipping")
                    missing_files += 1
                    valid = False
                    break
                
            if not valid:
                print(f"Skipping {puppet_id}: Invalid data in rounds")
                continue
            
            # Write combined file if we have data
            if len(puppet_data["rounds"]) > 0:
                out_path = output_path / f"{puppet_id}.json"
                
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(puppet_data, f, indent=2, ensure_ascii=False)
                
                successful_writes += 1
                tracker.mark_combined(puppet_id)
                
                if missing_files > 0:
                    print(f"Combined {puppet_id} ({len(puppet_data['rounds'])}/{expected_rounds} rounds, {missing_files} missing)")
            else:
                print(f"Skipping {puppet_id}: No valid round files found")
                
        except Exception as e:
            print(f"Error combining {puppet_id}: {e}")
    
    tracker.save_cache()
    
    end_time = time.time()
    print(f"Successfully combined {successful_writes}/{len(puppets_to_combine)} puppets")
    print(f"Total combination time: {end_time - start_time:.2f} seconds")
    
    total_combined = already_combined + successful_writes
    return len(puppet_ids), total_combined, len(puppet_ids) - total_combined, []


def stats_shell(base_dir, expected_rounds, output_prefix="experiment_stats"):
    """
    Ultra-fast stats using shell commands with incremental processing.
    """
    return incremental_stats_analysis(base_dir, expected_rounds, output_prefix, use_shell=True)

    
def summarize_success_failure_by_source(base_dir, successful_puppets, failed_puppets):
    base_path = Path(base_dir)
    tracker = ProcessingTracker(base_dir) 
    cache = tracker._load_cache()
    
    from collections import defaultdict
    source_to_success = defaultdict(set)
    source_to_failure = defaultdict(set)

    for pid in successful_puppets:
        source = cache["processed_puppets"].get(pid, {}).get("source", "Unknown")
        source_to_success[source].add(pid)

    for pid in failed_puppets:
        source = cache["processed_puppets"].get(pid, {}).get("source", "Unknown")
        source_to_failure[source].add(pid)

    all_sources = sorted(set(source_to_success) | set(source_to_failure))
    print("\n=== Source-based Success/Failure Summary ===")
    for source in all_sources:
        succ, fail = source_to_success[source], source_to_failure[source]
        total = len(succ) + len(fail)
        if total == 0:
            continue
        rate = 100 * len(succ) / total
        print(f"\nSource: {source}")
        print(f"  Total puppets: {total}")
        print(f"  Successful: {len(succ)}")
        print(f"  Failed: {len(fail)}")
        print(f"  Success rate: {rate:.1f}%")

    total_succ = sum(len(v) for v in source_to_success.values())
    total_fail = sum(len(v) for v in source_to_failure.values())
    total_all = total_succ + total_fail
    rate = 100 * total_succ / total_all if total_all else 0
    print("\nTOTAL across all sources:")
    print(f"  Total puppets: {total_all}")
    print(f"  Successful: {total_succ}")
    print(f"  Failed: {total_fail}")
    print(f"  Success rate: {rate:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Experiment processor with incremental processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Step 1: Incremental stats analysis (only checks new puppets)
  python fast_processor.py /path/to/experiments 30 --stats-only
  
  # Step 2: Incremental combination (uses cache, skips existing combined files)
  python fast_processor.py /path/to/experiments 30 --combine --output-dir ./combined_puppets
  
  # Force clean processing (ignores cache)
  python fast_processor.py /path/to/experiments 30 --stats-only --clean-cache
  
  # Check cache status
  python fast_processor.py /path/to/experiments 30 --show-cache
        """
    )
    
    parser.add_argument("directory", help="Directory containing experiment subdirectories")
    parser.add_argument("expected_rounds", type=int, help="Expected maximum round number")
    parser.add_argument("--combine", action="store_true", help="Combine JSON files for successful puppets")
    parser.add_argument("--output-dir", default='./combined_puppets', help="Directory to save combined files")
    parser.add_argument("--stats-only", action="store_true", help="Run stats analysis only")
    parser.add_argument("--puppet-args", action="store_true", help="Collect puppet args from output folders")
    parser.add_argument("--args-dir", default="../output/puppets", help="Collect puppet args from output folders")
    parser.add_argument("--success-list", help="Text file containing list of successful puppet IDs")
    parser.add_argument("--output-prefix", default="experiment_stats", help="Prefix for output files")
    parser.add_argument("--clean-cache", action="store_true", help="Ignore existing cache and reprocess everything")
    parser.add_argument("--show-cache", action="store_true", help="Show cache status and exit")
    
    args = parser.parse_args()
    
    if args.expected_rounds <= 0:
        print("Error: Expected rounds must be a positive integer.")
        sys.exit(1)
    
    # Handle cache operations
    if args.clean_cache:
        cache_file = Path(__file__).parent / ".processing_cache.json"
        if cache_file.exists():
            cache_file.unlink()
            print("Cleaned processing cache.")
        else:
            print("No cache file found.")
    
    if args.show_cache:
        tracker = ProcessingTracker(args.directory)
        successful = tracker.get_successful_puppets()
        failed = tracker.get_failed_puppets()
        uncombined = tracker.get_uncombined_successful_puppets()
        
        print(f"Cache status for: {args.directory}")
        print(f"Expected rounds: {tracker.cache.get('expected_rounds', 'Not set')}")
        print(f"Last stats run: {tracker.cache.get('last_stats_run', 'Never')}")
        print(f"Tracked puppets: {len(tracker.cache['processed_puppets'])}")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")
        print(f"  Successful but uncombined: {len(uncombined)}")
        return
    
    # Main processing
    if args.stats_only:
        all_puppets, all_successful, all_failed = stats_shell(args.directory, args.expected_rounds, args.output_prefix)
        if args.puppet_args:
            collect_puppet_args(args.directory, args.args_dir)
            summarize_puppet_distribution(args.directory)
        
        summarize_success_failure_by_source(args.directory, all_successful, all_failed)
    else:
        if args.combine:
            result = incremental_combine_files(
                args.directory,
                args.expected_rounds,
                args.output_dir,
                args.success_list
            )
    

if __name__ == "__main__":
    main()