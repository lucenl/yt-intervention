import os
import json
import argparse


def analyze_puppet_results(output_dir, homepage_threshold, upnext_threshold):
    """
    Analyze puppet results to identify successful and failed runs

    Args:
        output_dir: Directory containing puppet results
    """
    # Define output paths
    puppets_dir = os.path.join(output_dir, "puppets")
    exceptions_dir = os.path.join(output_dir, "exceptions")

    # Track results
    failed_puppets = []
    success_puppets = []
    partial_success = []

    # Check exceptions first
    exception_puppets = []
    if os.path.exists(exceptions_dir):
        exception_puppets = os.listdir(exceptions_dir)
        for puppet_id in exception_puppets:
            failed_puppets.append(
                {
                    "puppet_id": puppet_id,
                    "failure_type": "exception",
                    "details": "Found in exceptions directory",
                }
            )

    # Analyze puppet results
    if os.path.exists(puppets_dir):
        for puppet_file in os.listdir(puppets_dir):
            puppet_id = puppet_file

            # Skip if already in exceptions
            if puppet_id in exception_puppets:
                continue

            puppet_path = os.path.join(puppets_dir, puppet_file)

            try:
                with open(puppet_path, "r") as f:
                    data = json.load(f)

                # Look for specific failure patterns in actions
                actions = data.get("actions", [])

                # Initialize tracking variables
                has_training_end = False
                last_homepage_recs = []
                upnext_recs = []

                # Find relevant actions, focusing on the ones closest to training_end
                for i, action in enumerate(actions):
                    action_type = action.get("action")
                    params = action.get("params", [])

                    if action_type == "training_end":
                        has_training_end = True

                    if action_type == "get_homepage_recommendations":
                        # Reset each time we find this action - we only care about the last one
                        last_homepage_recs = params if params else []

                    if action_type == "get_upnext_recommendations":
                        # In your pattern, up-next recommendations happen before training_end
                        # and there's only one such action near the end
                        upnext_recs = params if params else []

                # Count the recommendations
                homepage_rec_count = (
                    len(last_homepage_recs) if last_homepage_recs else 0
                )
                upnext_rec_count = len(upnext_recs) if upnext_recs else 0

                # Determine success status - modified thresholds based on observed data
                if (
                    has_training_end
                    and homepage_rec_count >= homepage_threshold
                    and upnext_rec_count >= upnext_threshold
                ):
                    success_puppets.append(
                        {
                            "puppet_id": puppet_id,
                            "homepage_recs": homepage_rec_count,
                            "upnext_recs": upnext_rec_count,
                        }
                    )
                else:
                    # Determine failure type
                    failure_type = []
                    if not has_training_end:
                        failure_type.append("missing_training_end")
                    if homepage_rec_count < homepage_threshold:
                        failure_type.append(
                            f"insufficient_homepage_recs({homepage_rec_count})"
                        )
                    if upnext_rec_count < upnext_threshold:
                        failure_type.append(
                            f"insufficient_upnext_recs({upnext_rec_count})"
                        )

                    partial_success.append(
                        {
                            "puppet_id": puppet_id,
                            "failure_type": ", ".join(failure_type),
                            "homepage_recs": homepage_rec_count,
                            "upnext_recs": upnext_rec_count,
                        }
                    )

            except Exception as e:
                failed_puppets.append(
                    {
                        "puppet_id": puppet_id,
                        "failure_type": "parse_error",
                        "details": str(e),
                    }
                )

    # Generate summary report
    total_puppets = len(success_puppets) + len(partial_success) + len(failed_puppets)
    success_rate = (
        (len(success_puppets) / total_puppets * 100) if total_puppets > 0 else 0
    )

    print(f"\nAnalysis Results:")
    print(f"Total Puppets: {total_puppets}")
    print(f"Successful: {len(success_puppets)} ({success_rate:.1f}%)")
    print(f"Partial Success: {len(partial_success)}")
    print(f"Failed: {len(failed_puppets)}")

    # Write results to files
    output_dir = './output'
    write_list_to_file(
        success_puppets,
        os.path.join(output_dir, "successful_puppets.txt"),
        lambda p: f"{p['puppet_id']}: {p['homepage_recs']} homepage, {p['upnext_recs']} upnext",
    )
    write_list_to_file(
        partial_success,
        os.path.join(output_dir, "partial_success_puppets.txt"),
        lambda p: f"{p['puppet_id']}: {p['failure_type']}",
    )
    write_list_to_file(
        failed_puppets,
        os.path.join(output_dir, "failed_puppets.txt"),
        lambda p: f"{p['puppet_id']}: {p['failure_type']}",
    )

    # Create a simple list file with just successful puppet IDs
    with open(os.path.join(output_dir, "successful_puppet_ids.txt"), "w") as f:
        for puppet in success_puppets:
            f.write(f"{puppet['puppet_id']}\n")

    return success_puppets, partial_success, failed_puppets


def write_list_to_file(items, filename, formatter):
    """Write a list to a file using the provided formatter function"""
    with open(filename, "w") as f:
        for item in items:
            f.write(formatter(item) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze sock puppet results")
    parser.add_argument(
        "--output-dir", default="../output", help="Directory containing puppet results"
    )
    parser.add_argument(
        "--homepage-threshold",
        type=int,
        default=25,
        help="Minimum homepage recommendations required for success",
    )
    parser.add_argument(
        "--upnext-threshold",
        type=int,
        default=12,
        help="Minimum up-next recommendations required for success",
    )
    args = parser.parse_args()

    analyze_puppet_results(
        args.output_dir, args.homepage_threshold, args.upnext_threshold
    )