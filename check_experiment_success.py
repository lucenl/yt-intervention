#!/usr/bin/env python3
"""
Script to check the success rate of experiments based on round files.
"""

import os
import sys
import argparse
import re
from pathlib import Path


def check_experiment_success(base_dir, expected_rounds):
    """
    Check experiment success in subdirectories.
    
    Args:
        base_dir (str): Base directory containing experiment subdirectories
        expected_rounds (int): Expected maximum round number (Y)
    
    Returns:
        tuple: (total_subdirs, successful_subdirs, failed_subdirs, failed_list)
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Error: Directory '{base_dir}' does not exist.")
        return None
    
    if not base_path.is_dir():
        print(f"Error: '{base_dir}' is not a directory.")
        return None
    
    total_subdirs = 0
    successful_subdirs = 0
    failed_subdirs = 0
    failed_list = []
    
    # Pattern to match round files: round_X where X is a number
    round_pattern = re.compile(r'^round_(\d+)\.json$')
    
    print(f"Checking experiments in: {base_path.absolute()}")
    print(f"Expected rounds: 1 to {expected_rounds}")
    print("-" * 50)
    
    # Iterate through all subdirectories
    for subdir in sorted(base_path.iterdir()):
        if not subdir.is_dir():
            continue
            
        total_subdirs += 1
        
        # Find all round files in the subdirectory
        round_files = []
        for file_path in subdir.iterdir():
            if file_path.is_file():
                match = round_pattern.match(file_path.name)
                if match:
                    round_number = int(match.group(1))
                    round_files.append(round_number)
        
        # Check if we have all expected rounds
        round_files.sort()
        expected_rounds_set = set(range(1, expected_rounds + 1))
        actual_rounds_set = set(round_files)
        
        is_successful = (actual_rounds_set == expected_rounds_set)
        
        if is_successful:
            successful_subdirs += 1
            status = "✓ SUCCESS"
        else:
            failed_subdirs += 1
            status = "✗ FAILED"
            failed_list.append(subdir.name)
            
            # Show what's missing or extra
            missing_rounds = expected_rounds_set - actual_rounds_set
            extra_rounds = actual_rounds_set - expected_rounds_set
            
            details = []
            if missing_rounds:
                details.append(f"Missing: {sorted(missing_rounds)}")
            if extra_rounds:
                details.append(f"Extra: {sorted(extra_rounds)}")
            if details:
                status += f" ({', '.join(details)})"
        
        print(f"{subdir.name:30} {status}")
        
        # Show the rounds found (for debugging)
        if round_files:
            print(f"{'':30} Found rounds: {round_files}")
        else:
            print(f"{'':30} No round files found")
        print()
    
    return total_subdirs, successful_subdirs, failed_subdirs, failed_list


def main():
    parser = argparse.ArgumentParser(
        description="Check experiment success based on round files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiment_checker.py /path/to/experiments 10
  python experiment_checker.py ./experiments 5
  
This script checks each subdirectory for files named 'round_1', 'round_2', ..., 'round_Y'
where Y is the expected number of rounds.
        """
    )
    
    parser.add_argument(
        "directory",
        default="./experiments/intervention/debug-output/experiment_data",
        help="Directory containing experiment subdirectories"
    )
    
    parser.add_argument(
        "expected_rounds",
        default=3,
        type=int,
        help="Expected maximum round number (Y)"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output for each subdirectory"
    )
    
    args = parser.parse_args()
    
    if args.expected_rounds <= 0:
        print("Error: Expected rounds must be a positive integer.")
        sys.exit(1)
    
    result = check_experiment_success(args.directory, args.expected_rounds)
    
    if result is None:
        sys.exit(1)
    
    total_subdirs, successful_subdirs, failed_subdirs, failed_list = result
    
    # Print summary
    print("=" * 50)
    print("SUMMARY:")
    print(f"Total subdirectories: {total_subdirs}")
    print(f"Successful experiments: {successful_subdirs}")
    print(f"Failed experiments: {failed_subdirs}")
    
    if total_subdirs > 0:
        success_rate = (successful_subdirs / total_subdirs) * 100
        print(f"Success rate: {success_rate:.1f}%")
    
    if failed_list:
        print(f"\nFailed experiments:")
        for failed_exp in failed_list:
            print(f"  - {failed_exp}")


if __name__ == "__main__":
    main()