import os
import json
import random

def extract_harmful_percentage(puppet_id):
    """
    Given a puppet_id like "harmful_30_29,a7dc087c", extract the number after "harmful_" and before the second underscore.
    """
    if puppet_id.startswith("harmful_"):
        parts = puppet_id.split('_')
        if len(parts) >= 2:
            try:
                return int(parts[1].split(',')[0])  # Handle cases like "harmful_30,a7dc087c"
            except (ValueError, IndexError):
                return None
    return None

def process_puppet_file(puppet_path):
    """
    Process a JSON file for a puppet and extract homepage_recs, upnext_recs, and training list.
    Returns the recommendations, training list, and a success flag based on the criteria.
    """
    try:
        with open(puppet_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {puppet_path}: {e}")
        return None, None, False, False, []

    actions = data.get('actions', [])
    homepage_recs = []
    upnext_recs = []
    has_training_end = False

    for action in actions:
        action_type = action.get('action')
        params = action.get('params', [])

        if action_type == 'training_end':
            has_training_end = True

        if action_type == 'get_homepage_recommendations' and not homepage_recs:
            homepage_recs = params if params else []

        if action_type == 'get_upnext_recommendations' and not upnext_recs:
            upnext_recs = params if params else []

    # Check counts before trimming
    homepage_rec_count = len(homepage_recs)
    upnext_rec_count = len(upnext_recs)

    # Success criteria: at least 25 homepage and 12 up-next recommendations
    is_successful = homepage_rec_count >= 25 and upnext_rec_count >= 12

    # Trim recommendations after checking success
    if homepage_rec_count >= 25:
        homepage_recs = homepage_recs[:25]
    if upnext_rec_count >= 12:
        upnext_recs = upnext_recs[:12]

    # Extract training list from args
    training_list = data.get('args', {}).get('training', [])

    print(f"Get {homepage_rec_count} homepage recommendations and {upnext_rec_count} upnext recommendations for puppet {puppet_path}")
    return homepage_recs, upnext_recs, has_training_end, is_successful, training_list

def main():
    # Define directory and output paths
    puppets_dir = os.path.join('output', 'puppets')
    output_file = 'puppets.json'

    results = []
    exception_puppets = []  # List of puppet IDs to skip, if any

    # Get all puppet files from the output/puppets directory
    if not os.path.exists(puppets_dir):
        print(f"Directory {puppets_dir} does not exist.")
        return

    puppet_files = [f for f in os.listdir(puppets_dir) if os.path.isfile(os.path.join(puppets_dir, f))]
    stats = {}  # Dictionary to store count of successful puppets per harmful percentage

    for puppet_file in puppet_files:
        puppet_path = os.path.join(puppets_dir, puppet_file)
        puppet_id = os.path.splitext(puppet_file)[0]

        if puppet_id in exception_puppets:
            continue

        if not os.path.exists(puppet_path):
            print(f"File {puppet_path} does not exist.")
            continue

        processed = process_puppet_file(puppet_path)
        if processed[0] is None or not processed[2]:  # Check if training_end exists
            continue

        homepage_recs, upnext_recs, _, is_successful, training_list = processed
        if not is_successful:
            print(f"Puppet {puppet_id} does not meet success criteria (at least 25 homepage and 12 up-next recommendations). Skipping.")
            continue

        harmful_percentage = extract_harmful_percentage(puppet_id)
        if harmful_percentage is not None:
            stats[harmful_percentage] = stats.get(harmful_percentage, 0) + 1

        puppet_result = {
            "puppet_id": puppet_id,
            "harmful_percentage": harmful_percentage,
            "homepage_recs": homepage_recs,
            "upnext_recs": upnext_recs,
            "training": training_list  # Add training list to the result
        }
        results.append(puppet_result)

    # Print statistics
    print("\nStatistics of successful puppets by harmful percentage:")
    for percentage, count in sorted(stats.items()):
        print(f"Harmful Percentage {percentage}%: {count} puppets")

    # Determine the minimum number of puppets (e.g., 41 from your example)
    min_puppets = min(stats.values()) if stats else 0
    print(f"\nTrimming all harmful percentage groups to the minimum number of puppets: {min_puppets}")

    # Group results by harmful percentage and trim to min_puppets
    grouped_results = {}
    for result in results:
        percentage = result["harmful_percentage"]
        if percentage is not None:
            if percentage not in grouped_results:
                grouped_results[percentage] = []
            grouped_results[percentage].append(result)

    trimmed_results = []
    for percentage, puppets in grouped_results.items():
        if len(puppets) > min_puppets:
            # Randomly select min_puppets puppets
            trimmed_results.extend(random.sample(puppets, min_puppets))
        else:
            trimmed_results.extend(puppets)

    # Update stats with trimmed counts
    trimmed_stats = {percentage: min(min_puppets, len(puppets)) for percentage, puppets in grouped_results.items()}
    print("\nStatistics of trimmed puppets by harmful percentage:")
    for percentage, count in sorted(trimmed_stats.items()):
        print(f"Harmful Percentage {percentage}%: {count} puppets")

    # Save trimmed results to a single JSON file
    try:
        with open(output_file, 'w') as f:
            json.dump(trimmed_results, f, indent=4)
        print(f"\nResults saved to {output_file}")
    except Exception as e:
        print(f"Error writing output file: {e}")

if __name__ == "__main__":
    main()