import re
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

# Directory containing the log files
log_dir = "./success_intervention_logs"  # Adjust this path to where your log files are stored

# List of log files to process
log_files = [
    "harmful_30,d862ec0a_intervention_downrank_homepage.log",
    "harmful_50,33838ef7_intervention_downrank_homepage.log"
]

# Pattern to match "Found X harmful videos out of Y"
pattern = re.compile(r"Found (\d+) harmful videos out of (\d+)")

# Dictionary to store harmful videos per round for each experiment
experiment_data = {}

# Process each specified log file
for log_file in log_files:
    log_path = os.path.join(log_dir, log_file)
    
    if not os.path.exists(log_path):
        print(f"Log file {log_path} not found, skipping...")
        continue
    
    harmful_videos_per_round = []
    current_round = 0
    
    with open(log_path, 'r') as f:
        for line in f:
            # Check for round start to track round number
            if "Starting round" in line:
                current_round += 1
            # Look for harmful videos line
            match = pattern.search(line)
            if match:
                harmful_videos = int(match.group(1))
                total_videos = int(match.group(2))
                # Ensure rounds align (in case logs are out of order)
                while len(harmful_videos_per_round) < current_round:
                    harmful_videos_per_round.append(0)
                harmful_videos_per_round[current_round - 1] = harmful_videos
    
    # Use the log file name (without path) as the experiment name
    exp_name = log_file.replace("_intervention_downrank_homepage.log", "")
    experiment_data[exp_name] = harmful_videos_per_round

# Check if we have any data to plot
if not experiment_data:
    print("No data extracted from log files. Exiting.")
    exit()

# Plotting each experiment in a separate figure
for exp_name, harmful_videos in experiment_data.items():
    rounds = list(range(1, len(harmful_videos) + 1))
    
    # Create a new figure for each experiment
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, harmful_videos, marker='o', label=exp_name, color='blue')
    
    # Customize the plot
    plt.title(f"Number of Harmful Videos per Round\n({exp_name})")
    plt.xlabel("Round")
    plt.ylabel("Number of Harmful Videos")
    
    # Create a sparse x-axis by labeling every nth round
    num_rounds = len(rounds)
    if num_rounds > 5:  # Only sparsify if there are more than 5 rounds
        step = max(1, num_rounds // 5)  # Aim for ~5 labels
        sparse_ticks = list(range(1, num_rounds + 1, step))
        # Ensure the last round is included if it's not in the sparse ticks
        if sparse_ticks[-1] != num_rounds:
            sparse_ticks.append(num_rounds)
        plt.xticks(sparse_ticks)
    else:
        plt.xticks(rounds)
    
    # Force y-axis to show only integer ticks
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d'))  # Ensure no decimals (e.g., 0, not 0.0)
    
    # Ensure x-axis also shows integers without decimals
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%d'))
    
    plt.grid(True)
    plt.legend()
    
    # Save the plot to a file (unique filename for each experiment)
    output_filename = f"harmful_videos_per_round_{exp_name}.png"
    plt.savefig(output_filename)
    plt.close()  # Close the figure to free memory

print("Plots generated successfully.")