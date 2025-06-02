import json
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import numpy as np
import os
import random
import argparse

# Simulate base paths (user would replace these)
combined_dir = "./combined_puppets"
# Simulated paths
cache_path = ".processing_cache.json"

# Simulated loading from the processing cache
def load_cache(cache_path):
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Simulated reading all combined puppet JSONs
def load_combined_puppets(combined_dir):
    puppet_data = {}
    for file in Path(combined_dir).glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            puppet_data[file.stem] = json.load(f)
    return puppet_data

def visualize_harmful_exposure_trends(focus_filter):
    """
    Visualizes harmful exposure trends based on the processed puppets and their arguments.
    Focuses on the number of harmful videos across different rounds, grouped by focus, percentage, and intervention type.
    """
        
    # Load data
    cache = load_cache(cache_path)
    combined = load_combined_puppets(combined_dir)

    # Grouping structure: (focus, percentage, intervention) -> list of [num_harmful_videos for each round]
    grouped_trends = defaultdict(list)
    max_rounds = 30

    for puppet_id, info in cache["processed_puppets"].items():
        if info.get("status") != "success":
            continue
        args = info.get("arguments", {})
        if not args:
            continue
        focus = args.get("focus")
        perc = args.get("harmful_percentage")
        intervention = args.get("intervention_type")
        if not all([focus, perc is not None, intervention]):
            continue
        key = (focus, perc, intervention)
        puppet_json = combined.get(puppet_id)
        if not puppet_json:
            continue
        rounds = puppet_json.get("rounds", {})
        harm_per_round = []
        for i in range(1, max_rounds + 1):
            r = rounds.get(f"round_{i}", {})
            harm_count = r.get("num_harmful_videos", 0)
            harm_per_round.append(harm_count)
        grouped_trends[key].append(harm_per_round)

    # Filter keys by focus
    filtered_keys = {
        k: v for k, v in grouped_trends.items()
        if focus_filter == "both" or k[0] == focus_filter
    }

    # Determine min size only from filtered keys
    all_group_sizes = {k: len(v) for k, v in filtered_keys.items()}
    global_min_size = min(all_group_sizes.values())

    print(f"[{focus_filter}] Balancing all filtered settings to {global_min_size} puppets per condition")

    # Average using only filtered_keys
    averaged_trends = defaultdict(dict)

    random.seed(42)

    for (focus, perc, intervention), traces in filtered_keys.items():
        if len(traces) > global_min_size:
            sampled_traces = random.sample(traces, global_min_size)
        else:
            sampled_traces = traces
        avg_trace = np.mean(np.array(sampled_traces), axis=0)
        averaged_trends[(focus, perc)][intervention] = avg_trace

    # Sort values
    foci = sorted({k[0] for k in averaged_trends})
    percentages = sorted({k[1] for k in averaged_trends})
    interventions = sorted({i for d in averaged_trends.values() for i in d})

    # Plotting
    fig, axes = plt.subplots(len(foci), len(percentages), figsize=(4 * len(percentages), 4), sharex=True, sharey=True)

    # Always ensure axes is 2D array
    if len(foci) == 1 and len(percentages) == 1:
        axes = np.array([[axes]])
    elif len(foci) == 1:
        axes = axes[np.newaxis, :]  # shape (1, N)
    elif len(percentages) == 1:
        axes = axes[:, np.newaxis]  # shape (N, 1)


    for i, focus in enumerate(foci):
        for j, perc in enumerate(percentages):
            ax = axes[i][j]
            key = (focus, perc)
            if key in averaged_trends:
                for intervention in interventions:
                    if intervention in averaged_trends[key]:
                        ax.plot(range(1, max_rounds + 1), averaged_trends[key][intervention], label=intervention)
            ax.set_title(f"{focus} - {perc}%")
            if i == len(foci) - 1:
                ax.set_xlabel("Round")
            if j == 0:
                ax.set_ylabel("Avg # Harmful Videos")
            ax.grid(True)
            ax.legend()

    fig.suptitle("Harmful Exposure Trends", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"harmful_exposure_trends_{focus_filter}.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--focus-filter", choices=["homepage", "upnext", "both"], default="both",
        help="Which focus group(s) to include: homepage, upnext, or both"
    )
    args = parser.parse_args()
    visualize_harmful_exposure_trends(focus_filter=args.focus_filter)
    
    