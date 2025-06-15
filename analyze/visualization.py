import json
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import numpy as np
import os
import random
import argparse
from scipy.stats import ks_2samp
import string

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

def extract_harm_ratios_by_round(cache, combined, focus_filter, target_rounds=[1, 5, 15, 30], all_balanced=False):
    """
    Extract harmful ratios for specific rounds organized by (intervention, percentage)
    Returns: grouped_data[intervention][percentage][round] = [harm_ratios]
    """
    grouped_data = defaultdict(  # focus
        lambda: defaultdict(      # intervention
            lambda: defaultdict(  # perc
                lambda: defaultdict(list)  # round -> [ratio]
            )
        )
    )   
    
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
            """
            # Use calculate data from round 1 to round_num
            cumulative_harm = 0
            cumulative_recs = 0
                
            rounds_data = puppet_json.get("rounds", {})
            for r in range(1, round_num + 1):
                round_data = rounds_data.get(f"round_{r}", {})
                cumulative_harm += round_data.get("num_harmful_videos", 0)
                cumulative_recs += len(round_data.get("recommendations", []))
                
            total_harm = cumulative_harm
            total_recs = cumulative_recs
            """  
            rounds_data = puppet_json.get("rounds", {})
            round_data = rounds_data.get(f"round_{round_num}", {})
            total_harm = round_data.get("num_harmful_videos", 0)
            total_recs = len(round_data.get("recommendations", []))
                
            if total_recs == 0:
                continue
                
            harm_ratio = total_harm / total_recs
            grouped_data[focus][intervention][perc][round_num].append(harm_ratio)
    
    random.seed(42)
    if all_balanced:
        # Global balancing across all focuses
        all_group_sizes = {}
        for focus in grouped_data:
            for intervention in grouped_data[focus]:
                for perc in grouped_data[focus][intervention]:
                    for round_num in grouped_data[focus][intervention][perc]:
                        key = (focus, intervention, perc, round_num)
                        all_group_sizes[key] = len(grouped_data[focus][intervention][perc][round_num])
        
        if all_group_sizes:
            global_min_size = min(all_group_sizes.values())
            print(f"[{focus_filter}] Balancing all conditions to {global_min_size} puppets per condition")
            
            for focus in grouped_data:
                for intervention in grouped_data[focus]:
                    for perc in grouped_data[focus][intervention]:
                        for round_num in grouped_data[focus][intervention][perc]:
                            data = grouped_data[focus][intervention][perc][round_num]
                            if len(data) > global_min_size:
                                grouped_data[focus][intervention][perc][round_num] = random.sample(data, global_min_size)
            
            return grouped_data, global_min_size
    else:
        # Focus-specific balancing
        min_sizes = {}
        for focus in grouped_data:
            focus_group_sizes = {}
            for intervention in grouped_data[focus]:
                for perc in grouped_data[focus][intervention]:
                    for round_num in grouped_data[focus][intervention][perc]:
                        key = (intervention, perc, round_num)
                        focus_group_sizes[key] = len(grouped_data[focus][intervention][perc][round_num])
            
            if focus_group_sizes:
                focus_min_size = min(focus_group_sizes.values())
                min_sizes[focus] = focus_min_size
                print(f"[{focus_filter}] Balancing '{focus}' to {focus_min_size} puppets per condition")
                
                for intervention in grouped_data[focus]:
                    for perc in grouped_data[focus][intervention]:
                        for round_num in grouped_data[focus][intervention][perc]:
                            data = grouped_data[focus][intervention][perc][round_num]
                            if len(data) > focus_min_size:
                                grouped_data[focus][intervention][perc][round_num] = random.sample(data, focus_min_size)
        
        return grouped_data, min_sizes

def visualize_pre_post_intervention_cdf(cache, combined, focus_filter):
    # 1) original grouping: focus → intervention → perc → round → [ratios]
    raw_grouped, balanced_size = extract_harm_ratios_by_round(
        cache, combined, focus_filter, [1, 30], all_balanced=False
    )
    foci          = sorted(raw_grouped)                # e.g. ["homepage","upnext"]
    interventions = ["none","replace","downrank"]
    percentages   = sorted({
        perc
        for focus in foci
        for perc in raw_grouped[focus][interventions[0]]
    })

    # 2) build list of all (focus, intervention) combos
    combos = [(focus, interv) 
              for focus in foci 
              for interv in interventions]  # length = len(foci)*len(interventions) = 6

    # 3) setup figure: rows = #percentages, cols = #combos
    n_rows = len(percentages)  # 3
    n_cols = len(combos)       # 6
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3 * n_rows),
        sharex=True, sharey=True
    )
    # ensure axes is always 2D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    # 4) compute global x-limits
    all_vals = []
    for focus, interv in combos:
        for rnd in (1,30):
            for perc in percentages:
                all_vals.extend(raw_grouped[focus][interv][perc].get(rnd, []))
    xlim = (min(all_vals), max(all_vals)*1.05) if all_vals else (0,1)

    # 5) plot each cell
    letters = list(string.ascii_uppercase)  # at least 18 letters
    label_idx = 0

    for i_perc, perc in enumerate(percentages):
        for j_combo, (focus, interv) in enumerate(combos):
            ax = axes[i_perc][j_combo]

            # subplot letter
            ax.text(
                0.98, 0.98, letters[label_idx],
                transform=ax.transAxes, ha="right", va="top",
                fontsize=12, fontweight="bold"
            )
            label_idx += 1

            # get data
            r1  = raw_grouped[focus][interv][perc].get(1, [])
            r30 = raw_grouped[focus][interv][perc].get(30, [])

            # plot CDFs
            for color, (lab, data) in zip(("b","r"), [("Pre", r1), ("Post", r30)]):
                if not data:
                    continue
                vals = np.sort(data)
                y    = np.linspace(0,1,len(vals))
                ax.plot(vals, y, f"{color}-", lw=2, label=f"{lab} (n={len(data)})")

            # compute and show KS
            ks_stat = ks_test(r1, r30)
            ax.set_title(f"{focus} | {interv} | {perc}%\nKS: {ks_stat}", fontsize=8)

            # axis formatting
            ax.set_xlim(xlim)
            if i_perc == n_rows - 1:
                ax.set_xlabel("Harmful Video Percentage")
            if j_combo == 0:
                ax.set_ylabel("CDF")
            ax.grid(alpha=0.3)

    # 6) single legend & title
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, ["Pre","Post"],
        loc="lower center", bbox_to_anchor=(0.5,0.01), ncol=2
    )
    fig.suptitle(f"Pre vs Post Intervention CDF (n={balanced_size})", y=0.99)
    plt.tight_layout(rect=[0,0.03,1,0.98])
    plt.savefig("pre_post_cdf_3x6.png", dpi=300)
    plt.show()



def visualize_multi_round_cdf(cache, combined, focus_filter):
    """
    Visualize multiple rounds (1, 5, 15, 30) CDFs to understand temporal dynamics
    """
    # Extract data for multiple rounds
    target_rounds = [1, 5, 15, 30]
    grouped_data, balanced_size = extract_harm_ratios_by_round(cache, combined, focus_filter, target_rounds, all_balanced=False)
    
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
    
    legend_elements = []
    legend_labels = []
    legend_added = False
    
    for i, intervention in enumerate(interventions):
        for j, perc in enumerate(percentages):
            ax = axes[i][j]
            
            # Plot each round
            for round_num in target_rounds:
                round_data = grouped_data[intervention][perc].get(round_num, [])
                if round_data:
                    sorted_vals = np.sort(round_data)
                    yvals = np.linspace(0, 1, len(sorted_vals))
                    line = ax.plot(sorted_vals, yvals, color=round_colors[round_num], 
                           linewidth=2, label=f'Round {round_num} (n={len(round_data)})')[0]
                    
                    if not legend_added:
                        legend_elements.append(line)
                        legend_labels.append(f'Round {round_num}')
            
            if not legend_added:
                legend_added = True
            
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
                ax.set_xlabel("Harmful Video Percentage")
            if j == 0:  # Left column
                ax.set_ylabel("CDF")
            
            ax.grid(True, alpha=0.3)
            ax.set_xlim(x_range)
    
    title = f"Multi-Round CDF Evolution ({focus_filter}, n={balanced_size})" if focus_filter != "both" else "Multi-Round CDF Evolution"
    fig.suptitle(title, fontsize=16)
    
    if legend_elements:
        fig.legend(legend_elements, legend_labels, 
                  loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=len(target_rounds))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) 
    plt.savefig(f"multi_round_cdf_{focus_filter}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    
def visualize_harmful_exposure_trends(cache, combined, focus_filter, all_balanced=False):
    """
    Visualizes harmful exposure trends based on the processed puppets and their arguments.
    Focuses on the number of harmful videos across different rounds, grouped by focus, percentage, and intervention type.
    
    Args:
        all_balanced: If True, balance across all focuses. If False, balance within each focus separately.
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
        if focus_filter != "both" and focus != focus_filter:
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

    # Apply balancing strategy
    random.seed(42)
    
    if all_balanced:
        # Global balancing across all focuses
        all_group_sizes = {k: len(v) for k, v in grouped_trends.items()}
        if all_group_sizes:
            global_min_size = min(all_group_sizes.values())
            print(f"[{focus_filter}] Balancing all conditions to {global_min_size} puppets per condition")
            
            for key, traces in grouped_trends.items():
                if len(traces) > global_min_size:
                    grouped_trends[key] = random.sample(traces, global_min_size)
            min_size = global_min_size
        else:
            min_size = 0
    else:
        # Focus-specific balancing
        min_sizes = {}
        foci = sorted({k[0] for k in grouped_trends})
        
        for focus in foci:
            focus_keys = {k: v for k, v in grouped_trends.items() if k[0] == focus}
            if focus_keys:
                focus_min_size = min(len(v) for v in focus_keys.values())
                min_sizes[focus] = focus_min_size
                print(f"[{focus_filter}] Balancing '{focus}' to {focus_min_size} puppets per condition")
                
                for key, traces in focus_keys.items():
                    if len(traces) > focus_min_size:
                        grouped_trends[key] = random.sample(traces, focus_min_size)
        
        # For display purposes, use the minimum across all focuses
        min_size = min(min_sizes.values()) if min_sizes else 0

    # Compute averaged trends
    averaged_trends = defaultdict(dict)

    for (focus, perc, intervention), traces in grouped_trends.items():
        # Compute mean and 95CI
        data = np.array(traces)
        avg_trace = data.mean(axis=0)
        sem = data.std(axis=0, ddof=1) / np.sqrt(data.shape[0])
        ci_delta = 1.96 * sem  
        ci_lower = avg_trace - ci_delta
        ci_upper = avg_trace + ci_delta
      
        averaged_trends[(focus, perc)][intervention] = (avg_trace, ci_lower, ci_upper)

    # Sort values
    foci = sorted({k[0] for k in averaged_trends})
    percentages = sorted({k[1] for k in averaged_trends})
    interventions = ["none", "replace", "downrank"]  # Force specific order

    # Plotting
    fig, axes = plt.subplots(len(foci), len(percentages), figsize=(4 * len(percentages), 8), sharex=True, sharey=False)

    # Always ensure axes is 2D array
    if len(foci) == 1 and len(percentages) == 1:
        axes = np.array([[axes]])
    elif len(foci) == 1:
        axes = axes[np.newaxis, :]  # shape (1, N)
    elif len(percentages) == 1:
        axes = axes[:, np.newaxis]  # shape (N, 1)

    legend_elements = []
    legend_labels = []
    legend_added = False
    
    labels = ['A', 'B', 'C', 'D', 'E', 'F']
    for i, focus in enumerate(foci):
        for j, perc in enumerate(percentages):
            ax = axes[i][j]
            # subplot letter index
            idx = i * len(percentages) + j
            ax.text(
                0.98, 0.98,         # in axes fraction coordinates
                labels[idx],        # the letter
                transform=ax.transAxes,
                ha='right', va='top',
                fontsize=12,
                fontweight='bold'
            )
            key = (focus, perc)            
            if key in averaged_trends:
                for intervention in interventions:
                    if intervention in averaged_trends[key]:
                        avg_trace, ci_lower, ci_upper = averaged_trends[key][intervention]
                        x = np.arange(1, max_rounds + 1)
                        line = ax.plot(x, avg_trace, label=intervention)
                        ax.fill_between(x, ci_lower, ci_upper, alpha=0.3)
                        
                        if not legend_added:
                            legend_elements.append(line)
                            legend_labels.append(intervention)
                
                if not legend_added:
                    legend_added = True
            
            ax.set_title(f"{focus} - {perc}%")
            if i == len(foci) - 1:
                ax.set_xlabel("Round")
            if j == 0:
                ax.set_ylabel("Avg # Harmful Videos")
            ax.grid(True)

    balance_type = "Global" if all_balanced else "Focus-specific"
    title = f"Harmful Exposure Trends (n={min_sizes})"
    fig.suptitle(title, fontsize=14)

    handles, labels = axes[0][0].get_legend_handles_labels()

    if legend_elements:
        fig.legend(handles, labels, 
                  loc='lower center', bbox_to_anchor=(0.5, 0), ncol=len(interventions))
    
    for i in range(len(foci)):
        row_ymin = float('inf')
        row_ymax = float('-inf')
        for j in range(len(percentages)):
            ax = axes[i][j]
            y0, y1 = ax.get_ylim()
            if y0 < row_ymin: row_ymin = y0
            if y1 > row_ymax: row_ymax = y1

        for j in range(len(percentages)):
            axes[i][j].set_ylim(row_ymin, row_ymax)
            
    plt.tight_layout(rect=[0, 0.05, 1, 0.98]) 
    
    balance_suffix = "global" if all_balanced else "focus"
    plt.savefig(f"harmful_exposure_trends_{focus_filter}_{balance_suffix}.png", dpi=300)
    plt.show()


def visualize_harmful_percentage_trends(cache, combined, focus_filter, all_balanced=False):
    """
    Visualizes harmful exposure trends as percentages (ratios) based on the processed puppets and their arguments.
    Focuses on the percentage of harmful videos across different rounds, grouped by focus, percentage, and intervention type.
    
    Args:
        all_balanced: If True, balance across all focuses. If False, balance within each focus separately.
    """
    # Grouping structure: (focus, percentage, intervention) -> list of [harmful_ratios for each round]
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
        if focus_filter != "both" and focus != focus_filter:
            continue
        
        key = (focus, perc, intervention)
        puppet_json = combined.get(puppet_id)
        if not puppet_json:
            continue
        
        rounds = puppet_json.get("rounds", {})
        harm_ratio_per_round = []
        
        for i in range(1, max_rounds + 1):
            r = rounds.get(f"round_{i}", {})
            harm_count = r.get("num_harmful_videos", 0)
            recommendations = r.get("recommendations", [])
            total_recs = len(recommendations)
            
            # Calculate harmful ratio for this round
            if total_recs > 0:
                harm_ratio = harm_count / total_recs
            else:
                harm_ratio = 0.0
            
            harm_ratio_per_round.append(harm_ratio)
        
        grouped_trends[key].append(harm_ratio_per_round)

    # Apply balancing strategy
    random.seed(42)
    
    if all_balanced:
        # Global balancing across all focuses
        all_group_sizes = {k: len(v) for k, v in grouped_trends.items()}
        if all_group_sizes:
            global_min_size = min(all_group_sizes.values())
            print(f"[{focus_filter}] Balancing all conditions to {global_min_size} puppets per condition")
            
            for key, traces in grouped_trends.items():
                if len(traces) > global_min_size:
                    grouped_trends[key] = random.sample(traces, global_min_size)
            min_size = global_min_size
        else:
            min_size = 0
    else:
        # Focus-specific balancing
        min_sizes = {}
        foci = sorted({k[0] for k in grouped_trends})
        
        for focus in foci:
            focus_keys = {k: v for k, v in grouped_trends.items() if k[0] == focus}
            if focus_keys:
                focus_min_size = min(len(v) for v in focus_keys.values())
                min_sizes[focus] = focus_min_size
                print(f"[{focus_filter}] Balancing '{focus}' to {focus_min_size} puppets per condition")
                
                for key, traces in focus_keys.items():
                    if len(traces) > focus_min_size:
                        grouped_trends[key] = random.sample(traces, focus_min_size)
        
        # For display purposes, use the minimum across all focuses
        min_size = min(min_sizes.values()) if min_sizes else 0

    # Compute averaged trends
    averaged_trends = defaultdict(dict)

    for (focus, perc, intervention), traces in grouped_trends.items():
        # Compute mean and 95CI
        data = np.array(traces)
        avg_trace = data.mean(axis=0)
        sem = data.std(axis=0, ddof=1) / np.sqrt(data.shape[0])
        ci_delta = 1.96 * sem  
        ci_lower = avg_trace - ci_delta
        ci_upper = avg_trace + ci_delta
      
        averaged_trends[(focus, perc)][intervention] = (avg_trace, ci_lower, ci_upper)

    # Sort values
    foci = sorted({k[0] for k in averaged_trends})
    percentages = sorted({k[1] for k in averaged_trends})
    interventions = ["none", "replace", "downrank"]  # Force specific order

    # Plotting
    fig, axes = plt.subplots(len(foci), len(percentages), figsize=(4 * len(percentages), 8), sharex=True, sharey=False)

    # Always ensure axes is 2D array
    if len(foci) == 1 and len(percentages) == 1:
        axes = np.array([[axes]])
    elif len(foci) == 1:
        axes = axes[np.newaxis, :]  # shape (1, N)
    elif len(percentages) == 1:
        axes = axes[:, np.newaxis]  # shape (N, 1)

    legend_elements = []
    legend_labels = []
    legend_added = False
    
    labels = ['A', 'B', 'C', 'D', 'E', 'F']
    for i, focus in enumerate(foci):
        for j, perc in enumerate(percentages):
            ax = axes[i][j]
            # subplot letter index
            idx = i * len(percentages) + j
            ax.text(
                0.98, 0.98,         # in axes fraction coordinates
                labels[idx],        # the letter
                transform=ax.transAxes,
                ha='right', va='top',
                fontsize=12,
                fontweight='bold'
            )
            key = (focus, perc)            
            if key in averaged_trends:
                for intervention in interventions:
                    if intervention in averaged_trends[key]:
                        avg_trace, ci_lower, ci_upper = averaged_trends[key][intervention]
                        x = np.arange(1, max_rounds + 1)
                        line = ax.plot(x, avg_trace, label=intervention)
                        ax.fill_between(x, ci_lower, ci_upper, alpha=0.3)
                        
                        if not legend_added:
                            legend_elements.append(line)
                            legend_labels.append(intervention)
                
                if not legend_added:
                    legend_added = True
            
            ax.set_title(f"{focus} - {perc}%")
            if i == len(foci) - 1:
                ax.set_xlabel("Round")
            if j == 0:
                ax.set_ylabel("Avg Harmful Video Percentage")
            ax.grid(True)

    balance_type = "Global" if all_balanced else "Focus-specific"
    title = f"Harmful Percentage Trends (n={min_sizes})"
    fig.suptitle(title, fontsize=14)

    handles, labels = axes[0][0].get_legend_handles_labels()

    if legend_elements:
        fig.legend(handles, labels, 
                  loc='lower center', bbox_to_anchor=(0.5, 0), ncol=len(interventions))
    
    # Set consistent y-axis range for each row (focus)
    for i in range(len(foci)):
        row_ymin = float('inf')
        row_ymax = float('-inf')
        for j in range(len(percentages)):
            ax = axes[i][j]
            y0, y1 = ax.get_ylim()
            if y0 < row_ymin: row_ymin = y0
            if y1 > row_ymax: row_ymax = y1

        for j in range(len(percentages)):
            axes[i][j].set_ylim(row_ymin, row_ymax)
            
    plt.tight_layout(rect=[0, 0.05, 1, 0.98]) 
    
    balance_suffix = "global" if all_balanced else "focus"
    plt.savefig(f"harmful_percentage_trends_{focus_filter}_{balance_suffix}.png", dpi=300)
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
                ax.set_xlabel("Harmful Video Percentage")
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
    Now uses focus-specific balancing and x-axis ranges.
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

    # Focus-specific balancing (similar to trends function)
    min_sizes = {}
    foci = sorted({k[0] for k in grouped_ratios})
    
    random.seed(42)
    for focus in foci:
        focus_keys = {k: v for k, v in grouped_ratios.items() if k[0] == focus}
        if focus_keys:
            focus_min_size = min(len(v) for v in focus_keys.values())
            min_sizes[focus] = focus_min_size
            print(f"[{focus_filter}] Balancing '{focus}' to {focus_min_size} puppets per condition")
            
            # Apply balancing only to this focus
            for key, ratios in focus_keys.items():
                if len(ratios) > focus_min_size:
                    grouped_ratios[key] = random.sample(ratios, focus_min_size)

    # Get unique values for organizing subplots
    interventions = ["none", "replace", "downrank"]  # Force specific order
    percentages = sorted({k[2] for k in grouped_ratios})

    # Create subplots: rows for focus, columns for harmful percentages
    fig, axes = plt.subplots(len(foci), len(percentages), 
                            figsize=(5 * len(percentages), 4 * len(foci)), 
                            sharey=True, sharex=False)  # Don't share x-axis

    # Handle single subplot cases
    if len(foci) == 1 and len(percentages) == 1:
        axes = np.array([[axes]])
    elif len(foci) == 1:
        axes = axes[np.newaxis, :]  # shape (1, N)
    elif len(percentages) == 1:
        axes = axes[:, np.newaxis]  # shape (N, 1)

    legend_elements = []
    legend_labels = []
    legend_added = False

    # Determine x-axis range for each focus separately
    focus_x_ranges = {}
    for focus in foci:
        focus_values = []
        for key, values in grouped_ratios.items():
            if key[0] == focus:  # This key belongs to current focus
                focus_values.extend(values)
        
        if focus_values:
            focus_x_min = min(focus_values)
            focus_x_max = max(focus_values)
            # Add a small buffer to the max
            focus_x_ranges[focus] = (focus_x_min, focus_x_max * 1.05)
        else:
            focus_x_ranges[focus] = (0, 1)

    # Generate subplot labels
    labels = ['A', 'B', 'C', 'D', 'E', 'F']
    ks_results = []
    
    for i, focus in enumerate(foci):
        for j, perc in enumerate(percentages):
            ax = axes[i][j]
            
            # Add subplot letter
            idx = i * len(percentages) + j
            ax.text(
                0.98, 0.98,         # in axes fraction coordinates
                labels[idx],        # the letter
                transform=ax.transAxes,
                ha='right', va='top',
                fontsize=12,
                fontweight='bold'
            )
            
            intervention_data = {}
            # Plot each intervention as a separate line
            for intervention in interventions:
                key = (focus, intervention, perc)
                values = grouped_ratios.get(key, [])
                if not values:
                    continue
                    
                sorted_vals = np.sort(values)
                yvals = np.linspace(0, 1, len(sorted_vals))
                line = ax.plot(sorted_vals, yvals, linewidth=2)[0]
               
                if not legend_added:
                    legend_elements.append(line)
                    legend_labels.append(f"{intervention}")
            
            if not legend_added:
                legend_added = True
                
            # Perform KS tests between all pairs of interventions
            ks_comparisons = []
            comparison_pairs = [
                ("downrank", "replace"),
                ("downrank", "none"),
                ("replace", "none")
            ]
            
            for interv1, interv2 in comparison_pairs:
                data1 = intervention_data.get(interv1, [])
                data2 = intervention_data.get(interv2, [])
                
                if len(data1) > 0 and len(data2) > 0:
                    ks_stat, p_value = ks_test(data1, data2)
                    ks_comparisons.append(f"{interv1} vs {interv2}: {ks_stat}")
                    
                    # Store for table
                    ks_results.append({
                        'Focus': focus.title(),
                        'Harmful_Percentage': f"{perc}%",
                        'Comparison': f"{interv1.title()} vs {interv2.title()}",
                        'KS_Statistic': ks_stat.rstrip('*'),  # Remove significance markers for CSV
                        'p_value': p_value if not np.isnan(p_value) else "N/A",
                        'N': len(data1)  # Assuming balanced, both should be same length
                    })
            
            # Create title with KS comparisons
            title_lines = [f"{focus} - {perc}%"]
            # Split KS comparisons into two lines for better readability
            if len(ks_comparisons) >= 2:
                title_lines.append(ks_comparisons[0])  # downrank vs replace
                title_lines.append(f"{ks_comparisons[1]} | {ks_comparisons[2]}")  # other two
            elif len(ks_comparisons) == 1:
                title_lines.append(ks_comparisons[0])
            
            ax.set_title("\n".join(title_lines), fontsize=8)
            
           
            if i == len(foci) - 1:  # Bottom row
                ax.set_xlabel("Harmful Video Percentage")
            if j == 0:  # Left column
                ax.set_ylabel("CDF")
            
            ax.grid(True, alpha=0.3)
            # Use focus-specific x-axis range
            ax.set_xlim(focus_x_ranges[focus])

    # Create title with focus-specific sample sizes
    if len(min_sizes) == 1:
        sample_size_str = f"n={list(min_sizes.values())[0]}"
    else:
        sample_size_str = f"n={min_sizes}"
    
    if focus_filter == "both":
        title = f"CDF of Harmful Exposure by Harmful Percentage ({sample_size_str})"
    else:
        title = f"CDF of Harmful Exposure by Harmful Percentage ({focus_filter}, {sample_size_str})"
    fig.suptitle(title, fontsize=16)
    
    if legend_elements:
        fig.legend(legend_elements, legend_labels, 
                  loc='lower center', bbox_to_anchor=(0.5, 0), ncol=len(interventions))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])  
    plt.savefig(f"harmful_exposure_cdf_by_percentage_{focus_filter}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    generate_ks_comparison_table(ks_results, focus_filter)
    
    return ks_results
    
def generate_ks_comparison_table(ks_results, focus_filter):
    """
    Generate a comprehensive KS comparison table and save to CSV
    """
    if not ks_results:
        print("No KS results to generate table")
        return
    
    # Convert to DataFrame
    ks_df = pd.DataFrame(ks_results)
    
    # Sort by Focus, Harmful_Percentage, and Comparison for organized output
    ks_df = ks_df.sort_values(['Focus', 'Harmful_Percentage', 'Comparison'])
    
    # Save to CSV
    output_filename = f"ks_comparison_table_{focus_filter}.csv"
    ks_df.to_csv(output_filename, index=False)
    print(f"KS comparison table saved to {output_filename}")
    
    # Also create a formatted version for easy reading
    formatted_output = f"ks_comparison_formatted_{focus_filter}.txt"
    with open(formatted_output, 'w') as f:
        f.write("KS Test Comparison Results\n")
        f.write("=" * 50 + "\n\n")
        
        # Group by focus and percentage
        for focus in sorted(ks_df['Focus'].unique()):
            f.write(f"{focus} Results:\n")
            f.write("-" * 20 + "\n")
            
            focus_data = ks_df[ks_df['Focus'] == focus]
            for perc in sorted(focus_data['Harmful_Percentage'].unique()):
                perc_data = focus_data[focus_data['Harmful_Percentage'] == perc]
                f.write(f"\n{perc}:\n")
                
                for _, row in perc_data.iterrows():
                    p_val_str = f"p = {row['p_value']:.6f}" if row['p_value'] != "N/A" else "p = N/A"
                    f.write(f"  - {row['Comparison']}: KS = {row['KS_Statistic']}, {p_val_str}\n")
            
            f.write("\n")
    
    print(f"Formatted KS comparison saved to {formatted_output}")
    
    # Print summary to console
    print("\n=== KS Test Summary ===")
    for focus in sorted(ks_df['Focus'].unique()):
        print(f"\n{focus}:")
        focus_data = ks_df[ks_df['Focus'] == focus]
        for perc in sorted(focus_data['Harmful_Percentage'].unique()):
            print(f"  {perc}:")
            perc_data = focus_data[focus_data['Harmful_Percentage'] == perc]
            for _, row in perc_data.iterrows():
                p_val_str = f"p = {row['p_value']:.6f}" if row['p_value'] != "N/A" else "p = N/A"
                print(f"    - {row['Comparison']}: KS = {row['KS_Statistic']}, {p_val_str}")
    
    return ks_df
 
    
def main(plot_type, focus_filter):
    cache = load_cache(cache_path)
    combined = load_combined_puppets(combined_dir)
    if plot_type == "pre_post":
        visualize_pre_post_intervention_cdf(cache, combined, focus_filter)
    elif plot_type == "multi_round":
        visualize_multi_round_cdf(cache, combined, focus_filter)
    elif plot_type == "trends":
        visualize_harmful_exposure_trends(cache, combined, focus_filter, all_balanced=False)
    elif plot_type == "perc_trends":
        visualize_harmful_percentage_trends(cache, combined, focus_filter, all_balanced=False)
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
        "--plot-type", choices=["trends", "perc_trends", "cdf", "pre_post", "multi_round"], default="cdf",
        help="Which plot to generate: trends (line plot per round) or cdf (harmful ratio CDF)"
    )
    args = parser.parse_args()

    main(args.plot_type, args.focus_filter)