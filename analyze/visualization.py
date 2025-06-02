import json
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import numpy as np
import os
import random
import argparse

combined_dir = "./combined_puppets"
cache_path = ".processing_cache.json"
max_rounds = 30
HOME_RECS = 25
UPNEXT_RECS = 12

def load_cache(cache_path):
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_combined_puppets(combined_dir):
    puppet_data = {}
    files = list(Path(combined_dir).glob("*.json"))
    print(f"Loading {len(files)} combined puppet files...")
    
    for idx, file in enumerate(files):
        if idx % 100 == 0:
            print(f"Loading file {idx}/{len(files)}: {file.name}")
        with open(file, "r", encoding="utf-8") as f:
            puppet_data[file.stem] = json.load(f)
    return puppet_data

def visualize_harmful_exposure_trends(cache, combined, focus_filter):
    """
    Visualizes harmful exposure trends based on the processed puppets and their arguments.
    Focuses on the number of harmful videos across different rounds, grouped by focus, percentage, and intervention type.
    """
    # Grouping structure: (focus, percentage, intervention) -> list of [num_harmful_videos for each round]
    grouped_trends = defaultdict(list)

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

def visualize_cdf(cache, combined, focus_filter):
    """
    Visualizes the CDF of harmful exposure (ratio of harmful videos per puppet) grouped by focus and harmful_percentage.
    Each line in the plot corresponds to a different intervention type.
    """
    grouped_ratios = defaultdict(lambda: defaultdict(list))  # (focus, percentage) -> intervention -> [harm_ratios]

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
        if focus_filter != "both" and focus != focus_filter:
            continue

        puppet_json = combined.get(puppet_id)
        if not puppet_json:
            continue
        rounds = puppet_json.get("rounds", {})
        total_harm = 0
        total_recs = 0
        for r in rounds.values():
            total_harm += r.get("num_harmful_videos", 0)
            total_recs += HOME_RECS if focus == "homepage" else UPNEXT_RECS
        if total_recs == 0:
            continue
        harm_ratio = total_harm / total_recs
        grouped_ratios[(focus, perc)][intervention].append(harm_ratio)

    foci = sorted({k[0] for k in grouped_ratios})
    percentages = sorted({k[1] for k in grouped_ratios})
    interventions = sorted({i for d in grouped_ratios.values() for i in d})

    fig, axes = plt.subplots(1, len(foci), figsize=(6 * len(foci), 5), sharey=True)

    if len(foci) == 1:
        axes = [axes]  # Make it iterable

    for i, focus in enumerate(foci):
        ax = axes[i]
        for perc in percentages:
            key = (focus, perc)
            if key not in grouped_ratios:
                continue
            for intervention in interventions:
                values = grouped_ratios[key].get(intervention)
                if not values:
                    continue
                sorted_vals = np.sort(values)
                yvals = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)
                ax.plot(sorted_vals, yvals, label=f"{intervention} ({perc}%)")
        ax.set_title(f"{focus}")
        ax.set_xlabel("Harmful Video Ratio per Puppet")
        ax.set_ylabel("CDF")
        ax.grid(True)
        ax.legend()

    fig.suptitle("CDF of Harmful Exposure by Intervention", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"harmful_exposure_cdf_{focus_filter}.png", dpi=300)
    plt.show()


def main(plot_type, focus_filter):
    cache = load_cache(cache_path)
    combined = load_combined_puppets(combined_dir)
    if plot_type == "trends":
        visualize_harmful_exposure_trends(cache, combined, focus_filter)
    elif plot_type == "cdf":
        visualize_cdf(cache, combined, focus_filter)
    else:
        raise ValueError("Invalid plot type. Choose 'trends' or 'cdf'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--focus-filter", choices=["homepage", "upnext", "both"], default="both",
        help="Which focus group(s) to include: homepage, upnext, or both"
    )
    parser.add_argument(
        "--plot-type", choices=["trends", "cdf"], default="cdf",
        help="Which plot to generate: trends (line plot per round) or cdf (harmful ratio CDF)"
    )
    args = parser.parse_args()

    main(args.plot_type, args.focus_filter)