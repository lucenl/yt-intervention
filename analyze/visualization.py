import json
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import numpy as np
import os
import random
import argparse
from scipy.stats import ks_2samp

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

def ks_test(d1, d2):
    """Perform KS test and return formatted result"""
    if len(d1) == 0 or len(d2) == 0:
        return "N/A", "N/A"
    test = ks_2samp(d1, d2)
    significance = '**' if test.pvalue < 0.01 else '*' if test.pvalue < 0.05 else ''
    ks_stat = f'{test.statistic:.3f}{significance}'
    # p_val = f'{test.pvalue}'
    return ks_stat

def extract_harm_ratios_by_round(cache, combined, focus_filter, target_rounds=[1, 5, 15, 30]):
    """
    Extract harmful ratios for specific rounds organized by (intervention, percentage)
    Returns: grouped_data[intervention][percentage][round] = [harm_ratios]
    """
    grouped_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
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
            
        # Extract data for each target round
        for round_num in target_rounds:
            if round_num == 30:
                # Use total for round 30
                total = puppet_json.get("total", {})
                total_harm = total.get("num_harmful_videos", 0)
                total_recs = total.get("total_recommendations", 0)
            else:
                # Use specific round data
                round_data = puppet_json.get("rounds", {}).get(f"round_{round_num}", {})
                total_harm = round_data.get("num_harmful_videos", 0)
                total_recs = len(round_data.get("recommendations", []))
            
            if total_recs == 0:
                continue
                
            harm_ratio = total_harm / total_recs
            grouped_data[intervention][perc][round_num].append(harm_ratio)
    
    return grouped_data

def visualize_pre_post_intervention_cdf(cache, combined, focus_filter):
    """
    Visualize pre-intervention (round 1) vs post-intervention (round 30) CDFs
    with statistical significance tests
    """
    # Extract data for rounds 1 and 30
    grouped_data = extract_harm_ratios_by_round(cache, combined, focus_filter, [1, 30])
    
    # Get unique interventions and percentages
    interventions = ["none", "replace", "downrank"]
    percentages = sorted({perc for interv_data in grouped_data.values() 
                         for perc in interv_data.keys()})
    
    # Create subplot grid: rows for interventions, columns for percentages
    fig, axes = plt.subplots(len(interventions), len(percentages), 
                            figsize=(5 * len(percentages), 4 * len(interventions)), 
                            sharey=True)
    
    # Handle single subplot cases
    if len(interventions) == 1 and len(percentages) == 1:
        axes = np.array([[axes]])
    elif len(interventions) == 1:
        axes = axes[np.newaxis, :]
    elif len(percentages) == 1:
        axes = axes[:, np.newaxis]
    
    # Global x-axis range
    all_values = []
    for interv_data in grouped_data.values():
        for perc_data in interv_data.values():
            for round_data in perc_data.values():
                all_values.extend(round_data)
    
    if all_values:
        x_range = (min(all_values), max(all_values) * 1.05)
    else:
        x_range = (0, 1)
    
    for i, intervention in enumerate(interventions):
        for j, perc in enumerate(percentages):
            ax = axes[i][j]
            
            # Get data for this condition
            round_1_data = grouped_data[intervention][perc].get(1, [])
            round_30_data = grouped_data[intervention][perc].get(30, [])
            
            # Plot CDFs
            if round_1_data:
                sorted_vals = np.sort(round_1_data)
                yvals = np.linspace(0, 1, len(sorted_vals))
                ax.plot(sorted_vals, yvals, 'b-', linewidth=2, 
                       label=f'Pre (Round 1, n={len(round_1_data)})')
            
            if round_30_data:
                sorted_vals = np.sort(round_30_data)
                yvals = np.linspace(0, 1, len(sorted_vals))
                ax.plot(sorted_vals, yvals, 'r-', linewidth=2, 
                       label=f'Post (Round 30, n={len(round_30_data)})')
            
            # Statistical test
            ks_result = ks_test(round_1_data, round_30_data)
            
            # Set labels and title
            ax.set_title(f'{intervention} - {perc}%\nKS: {ks_result}', fontsize=10)
            
            if i == len(interventions) - 1:  # Bottom row
                ax.set_xlabel("Harmful Video Ratio")
            if j == 0:  # Left column
                ax.set_ylabel("CDF")
            
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            ax.set_xlim(x_range)
    
    title = f"Pre vs Post Intervention CDF Comparison ({focus_filter})" if focus_filter != "both" else "Pre vs Post Intervention CDF Comparison"
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"pre_post_intervention_cdf_{focus_filter}.png", dpi=300, bbox_inches='tight')
    plt.show()

def visualize_multi_round_cdf(cache, combined, focus_filter):
    """
    Visualize multiple rounds (1, 5, 15, 30) CDFs to understand temporal dynamics
    """
    # Extract data for multiple rounds
    target_rounds = [1, 5, 15, 30]
    grouped_data = extract_harm_ratios_by_round(cache, combined, focus_filter, target_rounds)
    
    # Get unique interventions and percentages
    interventions = ["none", "replace", "downrank"]
    percentages = sorted({perc for interv_data in grouped_data.values() 
                         for perc in interv_data.keys()})
    
    # Create subplot grid
    fig, axes = plt.subplots(len(interventions), len(percentages), 
                            figsize=(5 * len(percentages), 4 * len(interventions)), 
                            sharey=True)
    
    # Handle single subplot cases
    if len(interventions) == 1 and len(percentages) == 1:
        axes = np.array([[axes]])
    elif len(interventions) == 1:
        axes = axes[np.newaxis, :]
    elif len(percentages) == 1:
        axes = axes[:, np.newaxis]
    
    # Global x-axis range
    all_values = []
    for interv_data in grouped_data.values():
        for perc_data in interv_data.values():
            for round_data in perc_data.values():
                all_values.extend(round_data)
    
    if all_values:
        x_range = (min(all_values), max(all_values) * 1.05)
    else:
        x_range = (0, 1)
    
    # Color map for different rounds
    colors = ['blue', 'green', 'orange', 'red']
    round_colors = dict(zip(target_rounds, colors))
    
    for i, intervention in enumerate(interventions):
        for j, perc in enumerate(percentages):
            ax = axes[i][j]
            
            # Plot each round
            for round_num in target_rounds:
                round_data = grouped_data[intervention][perc].get(round_num, [])
                if round_data:
                    sorted_vals = np.sort(round_data)
                    yvals = np.linspace(0, 1, len(sorted_vals))
                    ax.plot(sorted_vals, yvals, color=round_colors[round_num], 
                           linewidth=2, label=f'Round {round_num} (n={len(round_data)})')
            
            # Statistical tests between consecutive rounds
            ks_results = []
            for k in range(len(target_rounds) - 1):
                r1, r2 = target_rounds[k], target_rounds[k + 1]
                data1 = grouped_data[intervention][perc].get(r1, [])
                data2 = grouped_data[intervention][perc].get(r2, [])
                ks_results.append(f"R{r1}-R{r2}: {ks_test(data1, data2)}")
            
            # Set labels and title
            title_text = f'{intervention} - {perc}%\n' + ' | '.join(ks_results[:2])
            if len(ks_results) > 2:
                title_text += f'\n{" | ".join(ks_results[2:])}'
            ax.set_title(title_text, fontsize=8)
            
            if i == len(interventions) - 1:  # Bottom row
                ax.set_xlabel("Harmful Video Ratio")
            if j == 0:  # Left column
                ax.set_ylabel("CDF")
            
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
            ax.set_xlim(x_range)
    
    title = f"Multi-Round CDF Evolution ({focus_filter})" if focus_filter != "both" else "Multi-Round CDF Evolution"
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"multi_round_cdf_{focus_filter}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    
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
                        ax.plot(range(1, max_rounds + 1), averaged_trends[key][intervention], label=f"intervention (n={global_min_size})")
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
    Visualizes the CDF of harmful exposure (ratio of harmful videos per puppet) grouped by intervention type.
    Each subplot focuses on one intervention, with lines representing different harmful percentages (0%, 25%, 50%).
    """
    grouped_ratios = defaultdict(list)  # (focus, intervention, percentage) -> [harm_ratios]

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
        total = puppet_json.get("total", {})
        total_harm = total.get("num_harmful_videos", 0)
        total_recs = total.get("total_recommendations", 0)
        if total_recs == 0:
            continue
        harm_ratio = total_harm / total_recs
        grouped_ratios[(focus, intervention, perc)].append(harm_ratio)

    # Determine minimum sample size across all conditions for balanced sampling
    all_group_sizes = {k: len(v) for k, v in grouped_ratios.items()}
    if all_group_sizes:
        global_min_size = min(all_group_sizes.values())
        print(f"[{focus_filter}] Balancing all conditions to {global_min_size} puppets per condition")
    else:
        global_min_size = 0

    # Balance sample sizes across conditions
    balanced_ratios = {}
    random.seed(42)

    for key, ratios in grouped_ratios.items():
        if len(ratios) > global_min_size:
            sampled_ratios = random.sample(ratios, global_min_size)
        else:
            sampled_ratios = ratios
        balanced_ratios[key] = sampled_ratios

    # Get unique values for organizing subplots (force intervention order)
    foci = sorted({k[0] for k in balanced_ratios})
    interventions = ["none", "replace", "downrank"]  # Force specific order
    percentages = sorted({k[2] for k in balanced_ratios})

    # Determine global x-axis range across all conditions
    all_values = []
    for key, values in balanced_ratios.items():
        all_values.extend(values)
    
    if all_values:
        global_x_min = min(all_values)
        global_x_max = max(all_values)
        # Add a small buffer to the max
        x_range = (global_x_min, global_x_max * 1.05)
    else:
        x_range = (0, 1)

    # Create subplots: rows for focus, columns for interventions
    fig, axes = plt.subplots(len(foci), len(interventions), 
                            figsize=(5 * len(interventions), 4 * len(foci)), 
                            sharey=True)

    # Handle single subplot cases
    if len(foci) == 1 and len(interventions) == 1:
        axes = np.array([[axes]])
    elif len(foci) == 1:
        axes = axes[np.newaxis, :]  # shape (1, N)
    elif len(interventions) == 1:
        axes = axes[:, np.newaxis]  # shape (N, 1)

    for i, focus in enumerate(foci):
        for j, intervention in enumerate(interventions):
            ax = axes[i][j]
            
            # Plot each percentage as a separate line
            for perc in percentages:
                key = (focus, intervention, perc)
                values = balanced_ratios.get(key, [])
                if not values:
                    continue
                    
                sorted_vals = np.sort(values)
                yvals = np.linspace(0, 1, len(sorted_vals))
                ax.plot(sorted_vals, yvals, 
                       label=f"{perc}% harmful (n={len(values)})", 
                        linewidth=2)
            
            # Set subplot title and labels
            if len(foci) > 1:
                ax.set_title(f"{focus} - {intervention}")
            else:
                ax.set_title(f"{intervention}")
            
            if i == len(foci) - 1:  # Bottom row
                ax.set_xlabel("Harmful Video Ratio per Puppet")
            if j == 0:  # Left column
                ax.set_ylabel("CDF")
            
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xlim(x_range)

    # Set overall title
    if focus_filter == "both":
        fig.suptitle("CDF of Harmful Exposure by Intervention Type", fontsize=16)
    else:
        fig.suptitle(f"CDF of Harmful Exposure by Intervention Type ({focus_filter})", fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"harmful_exposure_cdf_{focus_filter}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
def visualize_cdf_by_percentage(cache, combined, focus_filter):
    """
    Visualizes the CDF of harmful exposure (ratio of harmful videos per puppet) grouped by harmful percentage.
    Each subplot focuses on one harmful percentage, with lines representing different intervention types.
    """
    grouped_ratios = defaultdict(list)  # (focus, intervention, percentage) -> [harm_ratios]

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
        total = puppet_json.get("total", {})
        total_harm = total.get("num_harmful_videos", 0)
        total_recs = total.get("total_recommendations", 0)
        if total_recs == 0:
            continue
        harm_ratio = total_harm / total_recs
        grouped_ratios[(focus, intervention, perc)].append(harm_ratio)

    # Determine minimum sample size across all conditions for balanced sampling
    all_group_sizes = {k: len(v) for k, v in grouped_ratios.items()}
    if all_group_sizes:
        global_min_size = min(all_group_sizes.values())
        print(f"[{focus_filter}] Balancing all conditions to {global_min_size} puppets per condition")
    else:
        global_min_size = 0

    # Balance sample sizes across conditions
    balanced_ratios = {}
    random.seed(42)

    for key, ratios in grouped_ratios.items():
        if len(ratios) > global_min_size:
            sampled_ratios = random.sample(ratios, global_min_size)
        else:
            sampled_ratios = ratios
        balanced_ratios[key] = sampled_ratios

    # Determine global x-axis range across all conditions
    all_values = []
    for key, values in balanced_ratios.items():
        all_values.extend(values)
    
    if all_values:
        global_x_min = min(all_values)
        global_x_max = max(all_values)
        # Add a small buffer to the max
        x_range = (global_x_min, global_x_max * 1.05)
    else:
        x_range = (0, 1)

    # Get unique values for organizing subplots (group by percentage this time)
    foci = sorted({k[0] for k in balanced_ratios})
    interventions = ["none", "replace", "downrank"]  # Force specific order
    percentages = sorted({k[2] for k in balanced_ratios})

    # Create subplots: rows for focus, columns for harmful percentages
    fig, axes = plt.subplots(len(foci), len(percentages), 
                            figsize=(5 * len(percentages), 4 * len(foci)), 
                            sharey=True)

    # Handle single subplot cases
    if len(foci) == 1 and len(percentages) == 1:
        axes = np.array([[axes]])
    elif len(foci) == 1:
        axes = axes[np.newaxis, :]  # shape (1, N)
    elif len(percentages) == 1:
        axes = axes[:, np.newaxis]  # shape (N, 1)

    for i, focus in enumerate(foci):
        for j, perc in enumerate(percentages):
            ax = axes[i][j]
            
            # Plot each intervention as a separate line
            for intervention in interventions:
                key = (focus, intervention, perc)
                values = balanced_ratios.get(key, [])
                if not values:
                    continue
                    
                sorted_vals = np.sort(values)
                yvals = np.linspace(0, 1, len(sorted_vals))
                ax.plot(sorted_vals, yvals, 
                       label=f"{intervention} (n={len(values)})", 
                       linewidth=2)
            
            # Set subplot title and labels
            if len(foci) > 1:
                ax.set_title(f"{focus} - {perc}% harmful")
            else:
                ax.set_title(f"{perc}% harmful")
            
            if i == len(foci) - 1:  # Bottom row
                ax.set_xlabel("Harmful Video Ratio per Puppet")
            if j == 0:  # Left column
                ax.set_ylabel("CDF")
            
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xlim(x_range)

    # Set overall title
    if focus_filter == "both":
        fig.suptitle("CDF of Harmful Exposure by Harmful Percentage", fontsize=16)
    else:
        fig.suptitle(f"CDF of Harmful Exposure by Harmful Percentage ({focus_filter})", fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"harmful_exposure_cdf_by_percentage_{focus_filter}.png", dpi=300, bbox_inches='tight')
    plt.show()


def main(plot_type, focus_filter):
    cache = load_cache(cache_path)
    combined = load_combined_puppets(combined_dir)
    if plot_type == "pre_post":
        visualize_pre_post_intervention_cdf(cache, combined, focus_filter)
    elif plot_type == "multi_round":
        visualize_multi_round_cdf(cache, combined, focus_filter)
    elif plot_type == "trends":
        visualize_harmful_exposure_trends(cache, combined, focus_filter)
    elif plot_type == "cdf":
        # visualize_cdf(cache, combined, focus_filter)
        visualize_cdf_by_percentage(cache, combined, focus_filter)
    else:
        raise ValueError("Invalid plot type. Choose 'trends' or 'cdf'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--focus-filter", choices=["homepage", "upnext", "both"], default="both",
        help="Which focus group(s) to include: homepage, upnext, or both"
    )
    parser.add_argument(
        "--plot-type", choices=["trends", "cdf", "pre_post", "multi_round"], default="cdf",
        help="Which plot to generate: trends (line plot per round) or cdf (harmful ratio CDF)"
    )
    args = parser.parse_args()

    main(args.plot_type, args.focus_filter)