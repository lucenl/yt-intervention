import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import json
from pathlib import Path
import matplotlib.pyplot as plt
import string
import seaborn as sns
import os
import random
from collections import defaultdict
import argparse
import pickle
from datetime import datetime

# Category mapping
CATEGORY_MAPPING = {
    'SXL': 'Sexual',
    'HH': 'Hate', 
    'PH': 'Physical'
}

max_rounds = 30

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

def extract_and_save_plot_data(cache_path, combined_dir, output_path="plot_data_arrays.pkl"):
    """
    Extract all data needed for plotting and save as organized arrays
    This function should only be run when data changes
    
    How each line is calculated:
    1. For each puppet, extract category ratios for each round
    2. Group puppets by (focus, intervention, percentage, category)
    3. Apply focus-specific balancing (same sample size within each focus)
    4. For trends: compute mean + percentile-based CI across puppets for each round
    5. For CDF: collect all individual puppet values for rounds 1 and 30
    """
    print("=== EXTRACTING PLOT DATA ===")
    
    cache = load_cache(cache_path)
    combined = load_combined_puppets(combined_dir)
    
    # Extract raw data organized by conditions
    raw_data = defaultdict(list)  # (focus, intervention, percentage, category, round) -> [values]
    
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

        puppet_json = combined.get(puppet_id)
        if not puppet_json:
            continue
        
        rounds_data = puppet_json.get("rounds", {})
        
        # Extract data for each round
        for round_num in range(1, max_rounds + 1):
            round_data = rounds_data.get(f"round_{round_num}", {})
            recommendations = round_data.get("recommendations", [])
            total_recs = len(recommendations)
            
            if total_recs == 0:
                continue
                
            # Overall harmful ratio
            harmful_count = round_data.get("num_harmful_videos", 0)
            harmful_ratio = harmful_count / total_recs
            key = (focus, intervention, perc, "harmful", round_num)
            raw_data[key].append(harmful_ratio)
            
            # Category ratios - two types
            category_counts = round_data.get("harm_category_counts", {})
            for cat_code, cat_name in CATEGORY_MAPPING.items():
                cat_count = category_counts.get(cat_code, 0)
                
                # Type 1: Category ratio in total recommendations
                cat_ratio = cat_count / total_recs
                key = (focus, intervention, perc, cat_name.lower(), round_num)
                raw_data[key].append(cat_ratio)
                
                # Type 2: Category ratio within harmful content
                if harmful_count > 0:
                    cat_ratio_in_harm = cat_count / harmful_count
                else:
                    cat_ratio_in_harm = 0.0  # No harmful content, so 0 proportion
                key_in_harm = (focus, intervention, perc, f"{cat_name.lower()}_in_harm", round_num)
                raw_data[key_in_harm].append(cat_ratio_in_harm)
    
    print(f"Extracted data for {len(raw_data)} condition combinations")
    
    # Apply focus-specific balancing and organize for plotting
    plot_data = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'categories': ['harmful'] + [cat.lower() for cat in CATEGORY_MAPPING.values()],
            'categories_in_harm': [f"{cat.lower()}_in_harm" for cat in CATEGORY_MAPPING.values()],
            'all_categories': ['harmful'] + [cat.lower() for cat in CATEGORY_MAPPING.values()] + [f"{cat.lower()}_in_harm" for cat in CATEGORY_MAPPING.values()],
            'interventions': ["none", "replace", "downrank"],
            'percentages': [0, 25, 50],
            'rounds': list(range(1, max_rounds + 1)),
            'foci': ['homepage', 'upnext']
        },
        'cdf_data': {},      # For CDF plots: [focus][intervention][category][round] = array
        'trends_data': {},   # For trends plots: [focus][percentage][intervention][category] = array(rounds)
        'cumulative_ratios': {},  # For original CDF plots: [focus][intervention][percentage] = array of cumulative ratios
        'balance_info': {}   # Store balancing information
    }
    
    # Balance data by focus
    foci = ['homepage', 'upnext']
    categories = ['harmful'] + [cat.lower() for cat in CATEGORY_MAPPING.values()]
    categories_in_harm = [f"{cat.lower()}_in_harm" for cat in CATEGORY_MAPPING.values()]
    all_categories = categories + categories_in_harm
    interventions = ["none", "replace", "downrank"]
    percentages = [0, 25, 50]
    
    random.seed(42)
    
    for focus in foci:
        print(f"\nProcessing {focus}...")
        
        # Find minimum sample size for this focus
        focus_sizes = []
        for intervention in interventions:
            for perc in percentages:
                for category in all_categories:  # Now includes both types
                    for round_num in [1, 30]:  # Check key rounds
                        key = (focus, intervention, perc, category, round_num)
                        if key in raw_data:
                            focus_sizes.append(len(raw_data[key]))
        
        if not focus_sizes:
            continue
            
        min_size = min(focus_sizes)
        plot_data['balance_info'][focus] = min_size
        print(f"Balancing {focus} to {min_size} puppets per condition")
        
        # Initialize data structures for this focus
        plot_data['cdf_data'][focus] = {}
        plot_data['trends_data'][focus] = {}
        plot_data['cumulative_ratios'][focus] = {}  # Add cumulative ratios structure
        
        for intervention in interventions:
         
            plot_data['cdf_data'][focus][intervention] = {}

            for perc in percentages:
                if perc not in plot_data['cdf_data'][focus][intervention]:
                    plot_data['cdf_data'][focus][intervention][perc] = {}

                for category in all_categories:
                    plot_data['cdf_data'][focus][intervention][perc][category] = {}

                    for round_num in [1, 30]:
                        key = (focus, intervention, perc, category, round_num)
                        if key in raw_data:
                            data = raw_data[key]
                            if len(data) > min_size:
                                data = random.sample(data, min_size)
                            plot_data['cdf_data'][focus][intervention][perc][category][round_num] = np.array(data)

        # Trends data
        for perc in percentages:
            if perc not in plot_data['trends_data'][focus]:
                plot_data['trends_data'][focus][perc] = {}
            
            for intervention in interventions:
                if intervention not in plot_data['trends_data'][focus][perc]:
                    plot_data['trends_data'][focus][perc][intervention] = {}
                
                for category in all_categories:  # Now includes both ratio types
                    # Collect time series for all puppets in this condition
                    puppet_series = []
                    
                    # Get all puppets for this condition by looking at round 1 data
                    key_r1 = (focus, intervention, perc, category, 1)
                    if key_r1 not in raw_data:
                        continue
                        
                    n_puppets = min(len(raw_data[key_r1]), min_size)
                    puppet_indices = random.sample(range(len(raw_data[key_r1])), n_puppets)
                    
                    # For each selected puppet, build time series
                    for puppet_idx in puppet_indices:
                        series = []
                        for round_num in range(1, max_rounds + 1):
                            key = (focus, intervention, perc, category, round_num)
                            if key in raw_data and puppet_idx < len(raw_data[key]):
                                series.append(raw_data[key][puppet_idx])
                            else:
                                series.append(0.0)  # Fill missing with 0
                        puppet_series.append(series)
                    
                    if puppet_series:
                        # Convert to numpy array: shape (n_puppets, n_rounds)
                        puppet_array = np.array(puppet_series)
                        
                        # Compute statistics using consistent method across all conditions
                        mean_series = np.mean(puppet_array, axis=0)
                        std_series = np.std(puppet_array, axis=0, ddof=1)
                        n_puppets = puppet_array.shape[0]
                        se_series = std_series / np.sqrt(n_puppets)
                        
                        # Use simple truncation method (standard practice for bounded data)
                        # This ensures consistency across all conditions in the analysis
                        ci_lower = np.maximum(mean_series - 1.96 * se_series, 0.0)
                        ci_upper = np.minimum(mean_series + 1.96 * se_series, 1.0)
                        
                        plot_data['trends_data'][focus][perc][intervention][category] = {
                            'mean': mean_series,
                            'ci_lower': ci_lower,
                            'ci_upper': ci_upper,
                            'n_puppets': n_puppets,
                            'raw_data': puppet_array  # Keep raw data for additional analysis if needed
                        }

    # Calculate cumulative ratios for original CDF functionality
    print("\nCalculating cumulative ratios from 'total' data...")
    cumulative_data = defaultdict(list)  # (focus, intervention, perc, category) -> [ratios]
    
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

        puppet_json = combined.get(puppet_id)
        if not puppet_json:
            continue
        
        # Use the existing 'total' data - much simpler!
        total_data = puppet_json.get("total", {})
        total_harm = total_data.get("num_harmful_videos", 0)
        total_recs = total_data.get("total_recommendations", 0)
        
        if total_recs == 0:
            continue
            
        # Overall harmful ratio
        harmful_ratio = total_harm / total_recs
        cumulative_data[(focus, intervention, perc, 'harmful')].append(harmful_ratio)
        
        # Category ratios in total recommendations
        category_counts = total_data.get("harm_category_counts", {})
        for cat_code, cat_name in CATEGORY_MAPPING.items():
            cat_count = category_counts.get(cat_code, 0)
            cat_ratio = cat_count / total_recs
            cumulative_data[(focus, intervention, perc, cat_name.lower())].append(cat_ratio)
            
            # Category ratios within harmful content
            if total_harm > 0:
                cat_ratio_in_harm = cat_count / total_harm
            else:
                cat_ratio_in_harm = 0.0
            cumulative_data[(focus, intervention, perc, f"{cat_name.lower()}_in_harm")].append(cat_ratio_in_harm)
    
    # Apply focus-specific balancing to cumulative data
    for focus in foci:
        for intervention in interventions:
            if intervention not in plot_data['cumulative_ratios'][focus]:
                plot_data['cumulative_ratios'][focus][intervention] = {}
            
            for perc in percentages:
                if perc not in plot_data['cumulative_ratios'][focus][intervention]:
                    plot_data['cumulative_ratios'][focus][intervention][perc] = {}
                
                # Apply balancing to each category
                for category in all_categories + ['harmful']:  # Include 'harmful' category
                    key = (focus, intervention, perc, category)
                    if key in cumulative_data:
                        data = cumulative_data[key]
                        focus_min_size = plot_data['balance_info'].get(focus, len(data))
                        
                        if len(data) > focus_min_size:
                            balanced_data = random.sample(data, focus_min_size)
                        else:
                            balanced_data = data
                        
                        plot_data['cumulative_ratios'][focus][intervention][perc][category] = np.array(balanced_data)
    
    # Save plot data
    with open(output_path, 'wb') as f:
        pickle.dump(plot_data, f)
    
    print(f"\n=== PLOT DATA SAVED TO {output_path} ===")
    print("Data structure:")
    print(f"- Foci: {plot_data['metadata']['foci']}")
    print(f"- Categories (in total): {plot_data['metadata']['categories']}")
    print(f"- Categories (in harm): {plot_data['metadata']['categories_in_harm']}")
    print(f"- All categories: {len(plot_data['metadata']['all_categories'])} total")
    print(f"- Balance info: {plot_data['balance_info']}")
    print(f"- CDF data keys: {list(plot_data['cdf_data'].keys())}")
    print(f"- Trends data keys: {list(plot_data['trends_data'].keys())}")
    
    return plot_data

def load_plot_data(data_path="plot_data_arrays.pkl"):
    """
    Load pre-processed plot data
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Plot data file {data_path} not found. Run with --extract-data first.")
    
    with open(data_path, 'rb') as f:
        plot_data = pickle.load(f)
    
    print(f"Loaded plot data created at: {plot_data['metadata']['created_at']}")
    print(f"Balance info: {plot_data['balance_info']}")
    
    return plot_data

def ks_test(d1, d2):
    """Perform KS test and return formatted result"""
    if len(d1) == 0 or len(d2) == 0:
        return "N/A", "N/A"
    test = ks_2samp(d1, d2)
    significance = '**' if test.pvalue < 0.01 else '*' if test.pvalue < 0.05 else ''
    ks_stat = f'{test.statistic:.3f}{significance}'
    return ks_stat, test.pvalue

def plot_category_cdf(plot_data, focus_filter):
    """
    Plot category-specific CDFs using pre-processed arrays
    """
    print(f"=== PLOTTING CATEGORY CDF ({focus_filter}) ===")
    
    cdf_data = plot_data['cdf_data']
    balance_info = plot_data['balance_info']
    
    # Filter by focus
    if focus_filter == "both":
        foci = plot_data['metadata']['foci']
    else:
        foci = [focus_filter]
    
    interventions = plot_data['metadata']['interventions']
    categories = [cat for cat in plot_data['metadata']['categories'] if cat != 'harmful']  # Exclude overall harmful
    
    # Build combos
    combos = [(focus, interv) for focus in foci for interv in interventions]
    
    # Setup figure
    n_rows = len(categories)
    n_cols = len(combos)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3 * n_rows),
        sharex=True, sharey=True
    )
    
    # Ensure axes is always 2D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    # Compute global x-limits for each category
    category_xlims = {}
    for category in categories:
        all_vals = []
        for focus, interv in combos:
            if focus in cdf_data and interv in cdf_data[focus] and category in cdf_data[focus][interv]:
                for round_num in [1, 30]:
                    if round_num in cdf_data[focus][interv][category]:
                        all_vals.extend(cdf_data[focus][interv][category][round_num])
        category_xlims[category] = (min(all_vals), max(all_vals)*1.05) if all_vals else (0, 1)

    # Plot each cell
    letters = list(string.ascii_uppercase)
    label_idx = 0

    for i_cat, category in enumerate(categories):
        for j_combo, (focus, interv) in enumerate(combos):
            ax = axes[i_cat][j_combo]

            # Subplot letter
            ax.text(
                0.98, 0.98, letters[label_idx],
                transform=ax.transAxes, ha="right", va="top",
                fontsize=12, fontweight="bold"
            )
            label_idx += 1

            # Get pre and post data arrays
            r1 = cdf_data[focus][interv][category][1]
            r30 = cdf_data[focus][interv][category][30]



            # Plot CDFs
            for color, (lab, data) in zip(("b", "r"), [("Pre", r1), ("Post", r30)]):
                if len(data) == 0:
                    continue
                vals = np.sort(data)
                y = np.linspace(0, 1, len(vals))
                ax.plot(vals, y, f"{color}-", lw=2, label=f"{lab} (n={len(data)})")

            # Compute and show KS
            ks_stat, p_value = ks_test(r1, r30)
            ax.set_title(f"{focus} | {interv}\n{category.title()}\nKS: {ks_stat}", fontsize=8)

            # Axis formatting
            ax.set_xlim(category_xlims[category])
            if i_cat == n_rows - 1:
                ax.set_xlabel(f"{category.title()} Content Ratio")
            if j_combo == 0:
                ax.set_ylabel("CDF")
            ax.grid(alpha=0.3)

    # Legend and title
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            ["Pre", "Post"],
            loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2
        )
    fig.suptitle(f"Category-specific Pre vs Post Intervention CDF (n={balance_info})", y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(f"category_specific_cdf_{focus_filter}.png", dpi=300)
    plt.show()

def plot_category_trends(plot_data, focus_filter):
    """
    Plot category-specific trends using pre-processed arrays
    Shows category ratios relative to total recommendations over time
    """
    print(f"=== PLOTTING CATEGORY TRENDS ({focus_filter}) ===")
    
    trends_data = plot_data['trends_data']
    balance_info = plot_data['balance_info']
    
    # Filter by focus
    if focus_filter == "both":
        foci = plot_data['metadata']['foci']
    else:
        foci = [focus_filter]
    
    percentages = plot_data['metadata']['percentages']
    interventions = plot_data['metadata']['interventions']
    categories = [cat for cat in plot_data['metadata']['categories'] if cat != 'harmful']  # Exclude overall harmful
    
    # Plot for each category
    for category in categories:
        print(f"Plotting {category} trends...")
        
        # Create subplot grid for this category
        n_focus = len(foci)
        n_perc = len(percentages)
        
        fig, axes = plt.subplots(n_focus, n_perc, figsize=(4 * n_perc, 4 * n_focus), 
                                sharex=True, sharey=False)  # Don't share y-axis
        
        # Always ensure axes is 2D array
        if n_focus == 1 and n_perc == 1:
            axes = np.array([[axes]])
        elif n_focus == 1:
            axes = axes[np.newaxis, :]
        elif n_perc == 1:
            axes = axes[:, np.newaxis]

        legend_elements = []
        legend_labels = []
        legend_added = False
        
        labels = ['A', 'B', 'C', 'D', 'E', 'F']
        
        for i, focus in enumerate(foci):
            for j, perc in enumerate(percentages):
                ax = axes[i][j]
                
                # Subplot letter
                idx = i * n_perc + j
                if idx < len(labels):
                    ax.text(
                        0.98, 0.98, labels[idx],
                        transform=ax.transAxes,
                        ha='right', va='top',
                        fontsize=12,
                        fontweight='bold'
                    )
                
                # Plot each intervention
                for intervention in interventions:
                    if (focus in trends_data and perc in trends_data[focus] and 
                        intervention in trends_data[focus][perc] and 
                        category in trends_data[focus][perc][intervention]):
                        
                        data = trends_data[focus][perc][intervention][category]
                        mean_trace = data['mean']
                        ci_lower = data['ci_lower']
                        ci_upper = data['ci_upper']
                        
                        x = np.arange(1, max_rounds + 1)
                        line = ax.plot(x, mean_trace, label=intervention)
                        ax.fill_between(x, ci_lower, ci_upper, alpha=0.3)
                        
                        if not legend_added:
                            legend_elements.append(line[0])
                            legend_labels.append(intervention)
                
                if not legend_added:
                    legend_added = True
                
                ax.set_title(f"{focus} - {perc}%")
                if i == n_focus - 1:
                    ax.set_xlabel("Round")
                if j == 0:
                    ax.set_ylabel(f"Avg {category.title()} Ratio")
                ax.grid(True)

        # Set consistent y-axis range for each row (focus) independently
        # This allows better visualization of patterns within each focus
        for i in range(n_focus):
            row_ymin = float('inf')
            row_ymax = float('-inf')
            for j in range(n_perc):
                ax = axes[i][j]
                y0, y1 = ax.get_ylim()
                if y0 < row_ymin: row_ymin = y0
                if y1 > row_ymax: row_ymax = y1

            # Apply the row-specific range to all subplots in this row
            for j in range(n_perc):
                axes[i][j].set_ylim(row_ymin, row_ymax)

        # Title and legend for this category
        fig.suptitle(f"{category.title()} Content Trends (n={balance_info})", fontsize=14)
        
        if legend_elements:
            fig.legend(legend_elements, legend_labels, 
                      loc='lower center', bbox_to_anchor=(0.5, 0), ncol=len(interventions))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.98])
        plt.savefig(f"category_{category}_trends_{focus_filter}.png", dpi=300)
        plt.show()

def plot_category_in_harm_trends(plot_data, focus_filter):
    """
    Plot category-specific trends within harmful content using pre-processed arrays
    Shows what proportion of harmful content each category represents
    """
    print(f"=== PLOTTING CATEGORY IN HARM TRENDS ({focus_filter}) ===")
    
    trends_data = plot_data['trends_data']
    balance_info = plot_data['balance_info']
    
    # Filter by focus
    if focus_filter == "both":
        foci = plot_data['metadata']['foci']
    else:
        foci = [focus_filter]
    
    percentages = plot_data['metadata']['percentages']
    interventions = plot_data['metadata']['interventions']
    categories_in_harm = plot_data['metadata']['categories_in_harm']
    
    # Plot for each category
    for category_in_harm in categories_in_harm:
        category_name = category_in_harm.replace('_in_harm', '')
        print(f"Plotting {category_name} within harmful content trends...")
        
        # Create subplot grid for this category
        n_focus = len(foci)
        n_perc = len(percentages)
        
        fig, axes = plt.subplots(n_focus, n_perc, figsize=(4 * n_perc, 4 * n_focus), 
                                sharex=True, sharey=False)  # Don't share y-axis
        
        # Always ensure axes is 2D array
        if n_focus == 1 and n_perc == 1:
            axes = np.array([[axes]])
        elif n_focus == 1:
            axes = axes[np.newaxis, :]
        elif n_perc == 1:
            axes = axes[:, np.newaxis]

        legend_elements = []
        legend_labels = []
        legend_added = False
        
        labels = ['A', 'B', 'C', 'D', 'E', 'F']
        
        for i, focus in enumerate(foci):
            for j, perc in enumerate(percentages):
                ax = axes[i][j]
                
                # Subplot letter
                idx = i * n_perc + j
                if idx < len(labels):
                    ax.text(
                        0.98, 0.98, labels[idx],
                        transform=ax.transAxes,
                        ha='right', va='top',
                        fontsize=12,
                        fontweight='bold'
                    )
                
                # Plot each intervention
                for intervention in interventions:
                    if (focus in trends_data and perc in trends_data[focus] and 
                        intervention in trends_data[focus][perc] and 
                        category_in_harm in trends_data[focus][perc][intervention]):
                        
                        data = trends_data[focus][perc][intervention][category_in_harm]
                        mean_trace = data['mean']
                        ci_lower = data['ci_lower']
                        ci_upper = data['ci_upper']
                        
                        x = np.arange(1, max_rounds + 1)
                        line = ax.plot(x, mean_trace, label=intervention)
                        ax.fill_between(x, ci_lower, ci_upper, alpha=0.3)
                        
                        if not legend_added:
                            legend_elements.append(line[0])
                            legend_labels.append(intervention)
                
                if not legend_added:
                    legend_added = True
                
                ax.set_title(f"{focus} - {perc}%")
                if i == n_focus - 1:
                    ax.set_xlabel("Round")
                if j == 0:
                    ax.set_ylabel(f"{category_name.title()} Ratio in Harmful Content")
                ax.grid(True)

        # Set consistent y-axis range for each row (focus) independently
        # This allows better visualization of patterns within each focus
        for i in range(n_focus):
            row_ymin = float('inf')
            row_ymax = float('-inf')
            for j in range(n_perc):
                ax = axes[i][j]
                y0, y1 = ax.get_ylim()
                if y0 < row_ymin: row_ymin = y0
                if y1 > row_ymax: row_ymax = y1

            # Apply the row-specific range to all subplots in this row
            for j in range(n_perc):
                axes[i][j].set_ylim(row_ymin, row_ymax)

        # Title and legend for this category
        fig.suptitle(f"{category_name.title()} Proportion in Harmful Content (n={balance_info})", fontsize=14)
        
        if legend_elements:
            fig.legend(legend_elements, legend_labels, 
                      loc='lower center', bbox_to_anchor=(0.5, 0), ncol=len(interventions))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.98])
        plt.savefig(f"category_{category_name}_in_harm_trends_{focus_filter}.png", dpi=300)
        plt.show()

def plot_category_composition_bar(plot_data, focus_filter):
    """
    Plot pre-post bar chart comparison of category composition within harmful content
    Shows the proportion of each category within harmful content for round 1 vs round 30
    """
    print(f"=== PLOTTING CATEGORY COMPOSITION BAR CHART ({focus_filter}) ===")
    
    cdf_data = plot_data['cdf_data']  # Use cdf_data which has round 1 and 30 data
    balance_info = plot_data['balance_info']
    
    # Filter by focus
    if focus_filter == "both":
        foci = plot_data['metadata']['foci']
    else:
        foci = [focus_filter]
    
    interventions = plot_data['metadata']['interventions']
    categories_in_harm = plot_data['metadata']['categories_in_harm']
    
    # Setup figure
    n_rows = len(foci)
    n_cols = len(interventions)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        sharey=True
    )
    
    # Ensure axes is always 2D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    # Color map for categories
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue
    
    # Plot each cell
    letters = list(string.ascii_uppercase)
    label_idx = 0

    for i, focus in enumerate(foci):
        for j, intervention in enumerate(interventions):
            ax = axes[i][j]

            # Subplot letter
            ax.text(
                0.98, 0.98, letters[label_idx],
                transform=ax.transAxes, ha="right", va="top",
                fontsize=12, fontweight="bold"
            )
            label_idx += 1

            # Collect Pre (round 1) and Post (round 30) data
            pre_means = []
            post_means = []
            category_names = []
            
            for category_in_harm in categories_in_harm:
                category_name = category_in_harm.replace('_in_harm', '')
                
                if (focus in cdf_data and intervention in cdf_data[focus] and 
                    category_in_harm in cdf_data[focus][intervention]):
                    
                    r1_data = cdf_data[focus][intervention][category_in_harm].get(1, np.array([]))
                    r30_data = cdf_data[focus][intervention][category_in_harm].get(30, np.array([]))
                    
                    pre_means.append(np.mean(r1_data) if len(r1_data) > 0 else 0)
                    post_means.append(np.mean(r30_data) if len(r30_data) > 0 else 0)
                else:
                    pre_means.append(0)
                    post_means.append(0)
                
                category_names.append(category_name.title())

            # Normalize to ensure they sum to 1 (since they should represent proportions within harmful content)
            pre_total = sum(pre_means)
            post_total = sum(post_means)
            
            if pre_total > 0:
                pre_normalized = [m / pre_total for m in pre_means]
            else:
                pre_normalized = pre_means
                
            if post_total > 0:
                post_normalized = [m / post_total for m in post_means]
            else:
                post_normalized = post_means
            
            # Create grouped bar chart
            x = np.arange(len(category_names))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, pre_normalized, width, label='Pre (Round 1)', 
                          color=colors[:len(category_names)], alpha=0.7)
            bars2 = ax.bar(x + width/2, post_normalized, width, label='Post (Round 30)', 
                          color=colors[:len(category_names)], alpha=1.0)
            
            # Add value labels on bars
            for bar, value in zip(bars1, pre_normalized):
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{height:.2f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=7)
            
            for bar, value in zip(bars2, post_normalized):
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{height:.2f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=7)

            ax.set_title(f"{focus} - {intervention}")
            ax.set_ylabel('Proportion in Harmful Content')
            ax.set_xticks(x)
            ax.set_xticklabels(category_names)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1)

            # Add text showing sum verification
            ax.text(0.02, 0.94, f'Pre sum: {sum(pre_normalized):.2f}', 
                   transform=ax.transAxes, ha='left', va='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                   fontsize=7)
            ax.text(0.02, 0.88, f'Post sum: {sum(post_normalized):.2f}', 
                   transform=ax.transAxes, ha='left', va='top',
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5),
                   fontsize=7)

    fig.suptitle(f"Category Composition in Harmful Content: Pre vs Post (n={balance_info})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"category_composition_bar_{focus_filter}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_category_composition_area(plot_data, focus_filter):
    """
    Plot stacked area chart showing category composition evolution within harmful content
    Each subplot shows one focus+intervention combination
    """
    print(f"=== PLOTTING CATEGORY COMPOSITION AREA CHART ({focus_filter}) ===")
    
    trends_data = plot_data['trends_data']
    balance_info = plot_data['balance_info']
    
    # Filter by focus
    if focus_filter == "both":
        foci = plot_data['metadata']['foci']
    else:
        foci = [focus_filter]
    
    interventions = plot_data['metadata']['interventions']
    categories_in_harm = plot_data['metadata']['categories_in_harm']
    
    # Create subplot for each focus+intervention combination
    n_focus = len(foci)
    n_interv = len(interventions)
    
    fig, axes = plt.subplots(n_focus, n_interv, figsize=(5 * n_interv, 4 * n_focus), 
                            sharex=True, sharey=True)
    
    # Always ensure axes is 2D array
    if n_focus == 1 and n_interv == 1:
        axes = np.array([[axes]])
    elif n_focus == 1:
        axes = axes[np.newaxis, :]
    elif n_interv == 1:
        axes = axes[:, np.newaxis]

    # Color map for categories
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue
    
    labels = ['A', 'B', 'C', 'D', 'E', 'F']
    
    for i, focus in enumerate(foci):
        for j, intervention in enumerate(interventions):
            ax = axes[i][j]
            
            # Subplot letter
            idx = i * n_interv + j
            if idx < len(labels):
                ax.text(
                    0.98, 0.98, labels[idx],
                    transform=ax.transAxes,
                    ha='right', va='top',
                    fontsize=12,
                    fontweight='bold'
                )
            
            # Aggregate data across all percentages for this focus+intervention
            category_series = {}
            for category_in_harm in categories_in_harm:
                category_name = category_in_harm.replace('_in_harm', '')
                
                # Collect data from all percentages and average
                all_series = []
                for perc in plot_data['metadata']['percentages']:
                    if (focus in trends_data and perc in trends_data[focus] and 
                        intervention in trends_data[focus][perc] and 
                        category_in_harm in trends_data[focus][perc][intervention]):
                        
                        data = trends_data[focus][perc][intervention][category_in_harm]
                        all_series.append(data['mean'])
                
                if all_series:
                    # Average across percentages
                    category_series[category_name] = np.mean(all_series, axis=0)
                else:
                    category_series[category_name] = np.zeros(max_rounds)
            
            # Normalize each time point to sum to 1
            x = np.arange(1, max_rounds + 1)
            normalized_series = {}
            
            for round_idx in range(max_rounds):
                round_total = sum(series[round_idx] for series in category_series.values())
                if round_total > 0:
                    for category_name in category_series:
                        if category_name not in normalized_series:
                            normalized_series[category_name] = np.zeros(max_rounds)
                        normalized_series[category_name][round_idx] = category_series[category_name][round_idx] / round_total
                else:
                    for category_name in category_series:
                        if category_name not in normalized_series:
                            normalized_series[category_name] = np.zeros(max_rounds)
                        normalized_series[category_name][round_idx] = 0
            
            # Create stacked area plot
            bottom = np.zeros(max_rounds)
            for cat_idx, (category_name, series) in enumerate(normalized_series.items()):
                ax.fill_between(x, bottom, bottom + series, 
                               alpha=0.7, color=colors[cat_idx], 
                               label=f"{category_name.title()}")
                bottom += series
            
            ax.set_title(f"{focus} - {intervention}")
            if i == n_focus - 1:
                ax.set_xlabel("Round")
            if j == 0:
                ax.set_ylabel("Proportion in Harmful Content")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

    # Add legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, 
                  loc='lower center', bbox_to_anchor=(0.5, 0), ncol=len(categories_in_harm))
    
    fig.suptitle(f"Category Composition Evolution in Harmful Content (n={balance_info})", fontsize=14)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(f"category_composition_area_{focus_filter}.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_original_cdf_by_percentage(plot_data, focus_filter):
    """
    Original CDF visualization using cumulative harmful ratios (total_harm / total_recs)
    This matches the original visualize_cdf_by_percentage function's intention
    """
    print(f"=== PLOTTING ORIGINAL CDF BY PERCENTAGE ({focus_filter}) ===")
    
    cumulative_ratios = plot_data['cumulative_ratios']
    balance_info = plot_data['balance_info']
    
    # Filter by focus
    if focus_filter == "both":
        foci = plot_data['metadata']['foci']
    else:
        foci = [focus_filter]
    
    interventions = plot_data['metadata']['interventions']
    percentages = plot_data['metadata']['percentages']
    
    # Create subplots: rows for focus, columns for harmful percentages
    fig, axes = plt.subplots(len(foci), len(percentages), 
                            figsize=(5 * len(percentages), 4 * len(foci)), 
                            sharey=True, sharex=False)

    # Handle single subplot cases
    if len(foci) == 1 and len(percentages) == 1:
        axes = np.array([[axes]])
    elif len(foci) == 1:
        axes = axes[np.newaxis, :]
    elif len(percentages) == 1:
        axes = axes[:, np.newaxis]

    legend_elements = []
    legend_labels = []
    legend_added = False

    # Determine x-axis range for each focus separately
    focus_x_ranges = {}
    for focus in foci:
        focus_values = []
        if focus in cumulative_ratios:
            for intervention in interventions:
                if intervention in cumulative_ratios[focus]:
                    for perc in percentages:
                        if (perc in cumulative_ratios[focus][intervention] and
                            'harmful' in cumulative_ratios[focus][intervention][perc]):
                            focus_values.extend(cumulative_ratios[focus][intervention][perc]['harmful'])
        
        if focus_values:
            focus_x_min = min(focus_values)
            focus_x_max = max(focus_values)
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
            if idx < len(labels):
                ax.text(
                    0.98, 0.98, labels[idx],
                    transform=ax.transAxes,
                    ha='right', va='top',
                    fontsize=12,
                    fontweight='bold'
                )
            
            # Collect data for all interventions
            intervention_data = {}
            
            for intervention in interventions:
                if (focus in cumulative_ratios and 
                    intervention in cumulative_ratios[focus] and 
                    perc in cumulative_ratios[focus][intervention] and
                    'harmful' in cumulative_ratios[focus][intervention][perc]):
                    
                    values = cumulative_ratios[focus][intervention][perc]['harmful']
                    intervention_data[intervention] = values
                else:
                    intervention_data[intervention] = np.array([])
            
            # Plot each intervention as a separate line
            for intervention in interventions:
                values = intervention_data[intervention]
                if len(values) == 0:
                    continue
                    
                sorted_vals = np.sort(values)
                yvals = np.linspace(0, 1, len(sorted_vals))
                line = ax.plot(sorted_vals, yvals, linewidth=2, label=intervention)[0]
               
                if not legend_added:
                    legend_elements.append(line)
                    legend_labels.append(intervention)
            
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
                data1 = intervention_data.get(interv1, np.array([]))
                data2 = intervention_data.get(interv2, np.array([]))
                
                if len(data1) > 0 and len(data2) > 0:
                    ks_stat, p_value = ks_test(data1, data2)
                    ks_comparisons.append(f"{interv1} vs {interv2}: {ks_stat}")
                    
                    # Store for table
                    ks_results.append({
                        'Focus': focus.title(),
                        'Harmful_Percentage': f"{perc}%",
                        'Comparison': f"{interv1.title()} vs {interv2.title()}",
                        'KS_Statistic': ks_stat.rstrip('*'),
                        'p_value': p_value if not np.isnan(p_value) else "N/A",
                        'N': len(data1)
                    })
            
            # Create title with KS comparisons
            title_lines = [f"{focus} - {perc}%"]
            if len(ks_comparisons) >= 3:
                title_lines.append(ks_comparisons[0])  # downrank vs replace
                title_lines.append(f"{ks_comparisons[1]} | {ks_comparisons[2]}")  # other two
            elif len(ks_comparisons) > 0:
                title_lines.extend(ks_comparisons)
            
            ax.set_title("\n".join(title_lines), fontsize=8)
           
            if i == len(foci) - 1:  # Bottom row
                ax.set_xlabel("Harmful Video Percentage")
            if j == 0:  # Left column
                ax.set_ylabel("CDF")
            
            ax.grid(True, alpha=0.3)
            ax.set_xlim(focus_x_ranges[focus])

    # Create title with focus-specific sample sizes
    if len(balance_info) == 1:
        sample_size_str = f"n={list(balance_info.values())[0]}"
    else:
        sample_size_str = f"n={balance_info}"
    
    if focus_filter == "both":
        title = f"CDF of Harmful Exposure by Harmful Percentage ({sample_size_str})"
    else:
        title = f"CDF of Harmful Exposure by Harmful Percentage ({focus_filter}, {sample_size_str})"
    fig.suptitle(title, fontsize=16)
    
    if legend_elements:
        fig.legend(legend_elements, legend_labels, 
                  loc='lower center', bbox_to_anchor=(0.5, 0), ncol=len(interventions))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])  
    plt.savefig(f"original_cdf_by_percentage_{focus_filter}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Generate KS comparison table
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
    output_filename = f"enhanced_ks_comparison_table_{focus_filter}.csv"
    ks_df.to_csv(output_filename, index=False)
    print(f"KS comparison table saved to {output_filename}")
    
    # Also create a formatted version for easy reading
    formatted_output = f"enhanced_ks_comparison_formatted_{focus_filter}.txt"
    with open(formatted_output, 'w') as f:
        f.write("Enhanced KS Test Comparison Results\n")
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
    print("\n=== Enhanced KS Test Summary ===")
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
 
 
def plot_category_in_harm_cdf(plot_data, focus_filter):
    """
    Plot category-specific CDFs within harmful content using pre-processed arrays
    """
    print(f"=== PLOTTING CATEGORY IN HARM CDF ({focus_filter}) ===")
    
    cdf_data = plot_data['cdf_data']
    balance_info = plot_data['balance_info']
    
    # Filter by focus
    if focus_filter == "both":
        foci = plot_data['metadata']['foci']
    else:
        foci = [focus_filter]
    
    interventions = plot_data['metadata']['interventions']
    categories_in_harm = plot_data['metadata']['categories_in_harm']
    
    # Build combos
    combos = [(focus, interv) for focus in foci for interv in interventions]
    
    # Setup figure
    n_rows = len(categories_in_harm)
    n_cols = len(combos)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3 * n_rows),
        sharex=True, sharey=True
    )
    
    # Ensure axes is always 2D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    # Compute global x-limits for each category
    category_xlims = {}
    for category_in_harm in categories_in_harm:
        all_vals = []
        for focus, interv in combos:
            if (focus in cdf_data and interv in cdf_data[focus] and 
                category_in_harm in cdf_data[focus][interv]):
                for round_num in [1, 30]:
                    if round_num in cdf_data[focus][interv][category_in_harm]:
                        all_vals.extend(cdf_data[focus][interv][category_in_harm][round_num])
        category_xlims[category_in_harm] = (min(all_vals), max(all_vals)*1.05) if all_vals else (0, 1)

    # Plot each cell
    letters = list(string.ascii_uppercase)
    label_idx = 0

    for i_cat, category_in_harm in enumerate(categories_in_harm):
        category_name = category_in_harm.replace('_in_harm', '')
        for j_combo, (focus, interv) in enumerate(combos):
            ax = axes[i_cat][j_combo]

            # Subplot letter
            ax.text(
                0.98, 0.98, letters[label_idx],
                transform=ax.transAxes, ha="right", va="top",
                fontsize=12, fontweight="bold"
            )
            label_idx += 1

            # Get pre and post data arrays
            if (focus in cdf_data and interv in cdf_data[focus] and 
                category_in_harm in cdf_data[focus][interv]):
                r1 = cdf_data[focus][interv][category_in_harm].get(1, np.array([]))
                r30 = cdf_data[focus][interv][category_in_harm].get(30, np.array([]))
            else:
                r1, r30 = np.array([]), np.array([])

            # Plot CDFs
            for color, (lab, data) in zip(("b", "r"), [("Pre", r1), ("Post", r30)]):
                if len(data) == 0:
                    continue
                vals = np.sort(data)
                y = np.linspace(0, 1, len(vals))
                ax.plot(vals, y, f"{color}-", lw=2, label=f"{lab} (n={len(data)})")

            # Compute and show KS
            ks_stat, p_value = ks_test(r1, r30)
            ax.set_title(f"{focus} | {interv}\n{category_name.title()} in Harm\nKS: {ks_stat}", fontsize=8)

            # Axis formatting
            ax.set_xlim(category_xlims[category_in_harm])
            if i_cat == n_rows - 1:
                ax.set_xlabel(f"{category_name.title()} Ratio in Harmful Content")
            if j_combo == 0:
                ax.set_ylabel("CDF")
            ax.grid(alpha=0.3)

    # Legend and title
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            ["Pre", "Post"],
            loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2
        )
    fig.suptitle(f"Category Proportion in Harmful Content - Pre vs Post CDF (n={balance_info})", y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(f"category_in_harm_cdf_{focus_filter}.png", dpi=300)
    plt.show()
    """
    Plot category-specific trends using pre-processed arrays
    """
    print(f"=== PLOTTING CATEGORY TRENDS ({focus_filter}) ===")
    
    trends_data = plot_data['trends_data']
    balance_info = plot_data['balance_info']
    
    # Filter by focus
    if focus_filter == "both":
        foci = plot_data['metadata']['foci']
    else:
        foci = [focus_filter]
    
    percentages = plot_data['metadata']['percentages']
    interventions = plot_data['metadata']['interventions']
    categories = [cat for cat in plot_data['metadata']['categories'] if cat != 'harmful']
    
    # Plot for each category
    for category in categories:
        print(f"Plotting {category} trends...")
        
        # Create subplot grid for this category
        n_focus = len(foci)
        n_perc = len(percentages)
        
        fig, axes = plt.subplots(n_focus, n_perc, figsize=(4 * n_perc, 4 * n_focus), 
                                sharex=True, sharey=True)
        
        # Always ensure axes is 2D array
        if n_focus == 1 and n_perc == 1:
            axes = np.array([[axes]])
        elif n_focus == 1:
            axes = axes[np.newaxis, :]
        elif n_perc == 1:
            axes = axes[:, np.newaxis]

        legend_elements = []
        legend_labels = []
        legend_added = False
        
        labels = ['A', 'B', 'C', 'D', 'E', 'F']
        
        for i, focus in enumerate(foci):
            for j, perc in enumerate(percentages):
                ax = axes[i][j]
                
                # Subplot letter
                idx = i * n_perc + j
                if idx < len(labels):
                    ax.text(
                        0.98, 0.98, labels[idx],
                        transform=ax.transAxes,
                        ha='right', va='top',
                        fontsize=12,
                        fontweight='bold'
                    )
                
                # Plot each intervention
                for intervention in interventions:
                    if (focus in trends_data and perc in trends_data[focus] and 
                        intervention in trends_data[focus][perc] and 
                        category in trends_data[focus][perc][intervention]):
                        
                        data = trends_data[focus][perc][intervention][category]
                        mean_trace = data['mean']
                        ci_lower = data['ci_lower']
                        ci_upper = data['ci_upper']
                        
                        x = np.arange(1, max_rounds + 1)
                        line = ax.plot(x, mean_trace, label=intervention)
                        ax.fill_between(x, ci_lower, ci_upper, alpha=0.3)
                        
                        if not legend_added:
                            legend_elements.append(line[0])
                            legend_labels.append(intervention)
                
                if not legend_added:
                    legend_added = True
                
                ax.set_title(f"{focus} - {perc}%")
                if i == n_focus - 1:
                    ax.set_xlabel("Round")
                if j == 0:
                    ax.set_ylabel(f"Avg {category.title()} Ratio")
                ax.grid(True)

        # Set consistent y-axis range for each row (focus) independently
        # This allows better visualization of patterns within each focus
        for i in range(n_focus):
            row_ymin = float('inf')
            row_ymax = float('-inf')
            for j in range(n_perc):
                ax = axes[i][j]
                y0, y1 = ax.get_ylim()
                if y0 < row_ymin: row_ymin = y0
                if y1 > row_ymax: row_ymax = y1

            # Apply the row-specific range to all subplots in this row
            for j in range(n_perc):
                axes[i][j].set_ylim(row_ymin, row_ymax)

        # Title and legend for this category
        fig.suptitle(f"{category.title()} Content Trends (n={balance_info})", fontsize=14)
        
        if legend_elements:
            fig.legend(legend_elements, legend_labels, 
                      loc='lower center', bbox_to_anchor=(0.5, 0), ncol=len(interventions))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.98])
        plt.savefig(f"category_{category}_trends_{focus_filter}.png", dpi=300)
        plt.show()

def plot_intervention_comparison_cdf(plot_data, focus_filter):
    """
    Plot CDF comparison between different interventions (not pre-post)
    Shows intervention effects by comparing final distributions (Round 30)
    """
    print(f"=== PLOTTING INTERVENTION COMPARISON CDF ({focus_filter}) ===")
    
    cdf_data = plot_data['cdf_data']
    balance_info = plot_data['balance_info']
    
    # Filter by focus
    if focus_filter == "both":
        foci = plot_data['metadata']['foci']
    else:
        foci = [focus_filter]
    
    interventions = plot_data['metadata']['interventions']
    categories = [cat for cat in plot_data['metadata']['categories'] if cat != 'harmful']
    
    # Setup figure: rows for categories, columns for foci
    n_rows = len(categories)
    n_cols = len(foci)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6 * n_cols, 4 * n_rows),
        sharex=False, sharey=True
    )
    
    # Ensure axes is always 2D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    # Colors for interventions
    colors = {'none': '#1f77b4', 'replace': '#ff7f0e', 'downrank': '#2ca02c'}
    
    # Plot each cell
    letters = list(string.ascii_uppercase)
    label_idx = 0
    
    # Store KS results for summary
    ks_results = []

    for i_cat, category in enumerate(categories):
        for j_focus, focus in enumerate(foci):
            ax = axes[i_cat][j_focus]

            # Subplot letter
            ax.text(
                0.98, 0.98, letters[label_idx],
                transform=ax.transAxes, ha="right", va="top",
                fontsize=12, fontweight="bold"
            )
            label_idx += 1

            # Collect data for all interventions (Round 30 only)
            intervention_data = {}
            
            for intervention in interventions:
                if (focus in cdf_data and intervention in cdf_data[focus] and 
                    category in cdf_data[focus][intervention]):
                    
                    # Use Round 30 data (post-intervention)
                    r30_data = cdf_data[focus][intervention][category].get(30, np.array([]))
                    intervention_data[intervention] = r30_data
                else:
                    intervention_data[intervention] = np.array([])

            # Plot CDFs for each intervention
            for intervention in interventions:
                data = intervention_data[intervention]
                if len(data) == 0:
                    continue
                    
                sorted_vals = np.sort(data)
                yvals = np.linspace(0, 1, len(sorted_vals))
                ax.plot(sorted_vals, yvals, color=colors[intervention], 
                       linewidth=2, label=f"{intervention} (n={len(data)})")

            # Perform KS tests between interventions
            ks_comparisons = []
            comparison_pairs = [
                ("replace", "none"),
                ("downrank", "none"), 
                ("downrank", "replace")
            ]
            
            for interv1, interv2 in comparison_pairs:
                data1 = intervention_data.get(interv1, np.array([]))
                data2 = intervention_data.get(interv2, np.array([]))
                
                if len(data1) > 0 and len(data2) > 0:
                    test = ks_2samp(data1, data2)
                    significance = '**' if test.pvalue < 0.01 else '*' if test.pvalue < 0.05 else ''
                    ks_stat = f'{test.statistic:.3f}{significance}'
                    ks_comparisons.append(f"{interv1} vs {interv2}: {ks_stat}")
                    
                    # Store results
                    ks_results.append({
                        'Category': category.title(),
                        'Focus': focus.title(),
                        'Comparison': f"{interv1.title()} vs {interv2.title()}",
                        'KS_Statistic': test.statistic,
                        'p_value': test.pvalue,
                        'Significant': test.pvalue < 0.05,
                        'N1': len(data1),
                        'N2': len(data2)
                    })

            # Create title with KS comparisons
            title_lines = [f"{focus} - {category.title()}"]
            if len(ks_comparisons) >= 3:
                title_lines.append(ks_comparisons[0])  # replace vs none
                title_lines.append(ks_comparisons[1])  # downrank vs none  
                title_lines.append(ks_comparisons[2])  # downrank vs replace
            elif len(ks_comparisons) > 0:
                title_lines.extend(ks_comparisons)
            
            ax.set_title("\n".join(title_lines), fontsize=9)

            # Axis formatting
            if i_cat == n_rows - 1:
                ax.set_xlabel(f"{category.title()} Content Ratio")
            if j_focus == 0:
                ax.set_ylabel("CDF")
            ax.grid(alpha=0.3)
            ax.legend()

    fig.suptitle(f"Intervention Comparison: Final Distributions (Round 30) - n={balance_info}", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"intervention_comparison_cdf_{focus_filter}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return ks_results

def plot_intervention_effect_summary(ks_results):
    """
    Create a summary heatmap of intervention effects
    """
    import pandas as pd
    import seaborn as sns
    
    # Convert to DataFrame
    df = pd.DataFrame(ks_results)
    
    # Create pivot table for heatmap
    pivot_data = df.pivot_table(
        values='KS_Statistic', 
        index=['Category', 'Focus'], 
        columns='Comparison', 
        fill_value=0
    )
    
    # Create significance mask
    pivot_sig = df.pivot_table(
        values='Significant', 
        index=['Category', 'Focus'], 
        columns='Comparison', 
        fill_value=False
    )
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create custom annotations with significance
    annot_data = pivot_data.copy()
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            val = pivot_data.iloc[i, j]
            sig = pivot_sig.iloc[i, j]
            if val > 0:
                annot_data.iloc[i, j] = f"{val:.3f}{'*' if sig else ''}"
            else:
                annot_data.iloc[i, j] = ""
    
    sns.heatmap(pivot_data, annot=annot_data, fmt='', cmap='RdYlBu_r', 
                center=0, ax=ax, cbar_kws={'label': 'KS Statistic'})
    
    ax.set_title("Intervention Effect Summary (KS Statistics)\n* indicates p < 0.05")
    ax.set_xlabel("Intervention Comparison")
    ax.set_ylabel("Category × Focus")
    
    plt.tight_layout()
    plt.savefig("intervention_effect_summary.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return pivot_data

def analyze_intervention_effects(plot_data, focus_filter="both"):
    """
    Complete intervention comparison analysis
    """
    # Generate CDF comparison
    ks_results = plot_intervention_comparison_cdf(plot_data, focus_filter)
    
    # Generate summary heatmap
    if ks_results:
        summary_data = plot_intervention_effect_summary(ks_results)
        
        # Print key findings
        print("\n=== KEY FINDINGS ===")
        df = pd.DataFrame(ks_results)
        
        # Find significant effects
        sig_effects = df[df['Significant'] == True]
        if len(sig_effects) > 0:
            print("Significant intervention effects:")
            for _, row in sig_effects.iterrows():
                print(f"- {row['Category']} ({row['Focus']}): {row['Comparison']} - KS={row['KS_Statistic']:.3f}, p={row['p_value']:.4f}")
        else:
            print("No statistically significant intervention effects found.")
        
        # Summary by comparison type
        print("\nSummary by comparison type:")
        for comp in df['Comparison'].unique():
            comp_data = df[df['Comparison'] == comp]
            sig_count = sum(comp_data['Significant'])
            total_count = len(comp_data)
            avg_ks = comp_data['KS_Statistic'].mean()
            print(f"- {comp}: {sig_count}/{total_count} significant, avg KS = {avg_ks:.3f}")
    
    return ks_results

def plot_all_category_cdf_by_condition(plot_data):
    """
    For each category (excluding 'harmful'), plot category-specific CDFs using pre-processed arrays
    Now grouped by (focus, intervention, percentage) — 3x6 grid per category
    """
    cdf_data = plot_data['cdf_data']
    balance_info = plot_data['balance_info']
    
    foci = plot_data['metadata']['foci']
    interventions = plot_data['metadata']['interventions']
    percentages = plot_data['metadata']['percentages']
    
    # 获取所有类别，去掉'harmful'
    all_categories = ['sexual', 'hate', 'physical']
    
    for category in all_categories:
        # Setup figure: 3 rows (percentages), 6 columns (3 interventions x 2 foci)
        fig, axes = plt.subplots(3, 6, figsize=(24, 12), sharex=True, sharey=True)
        axes = axes.reshape(3, 6)

        # Compute focus-specific y-limits
        focus_y_range = {}
        for focus in foci:
            y_vals = []
            for intervention in interventions:
                for perc in percentages:
                    for r in [1, 30]:
                        values = cdf_data[focus][intervention][perc][category].get(r, [])
                        y_vals.extend(values)
            focus_y_range[focus] = (0, 1) if y_vals else (0, 1)
        
        subplot_idx = 0
        letters = list(string.ascii_uppercase)

        for i, focus in enumerate(foci):
            for j, intervention in enumerate(interventions):
                for k, perc in enumerate(percentages):
                    ax = axes[k][i * 3 + j]
                    data_r1 = cdf_data[focus][intervention][perc][category].get(1, [])
                    data_r30 = cdf_data[focus][intervention][perc][category].get(30, [])

                    # Plot CDFs
                    for color, (label, d) in zip(('b', 'r'), [('Pre', data_r1), ('Post', data_r30)]):
                        if len(d) > 0:
                            vals = np.sort(d)
                            y = np.linspace(0, 1, len(vals))
                            ax.plot(vals, y, f"{color}-", lw=2, label=label)

                    # Compute and show KS
                    ks_stat, _ = ks_test(data_r1, data_r30)
                    ax.set_title(f"{focus} | {intervention} | {perc}%\nKS: {ks_stat}", fontsize=8)

                    ax.set_ylim(focus_y_range[focus])
                    if k == 2:
                        ax.set_xlabel(f"{category.title()} Ratio")
                    if i * 3 + j == 0:
                        ax.set_ylabel("CDF")
                    ax.grid(alpha=0.3)
                    ax.text(
                        0.98, 0.98, letters[subplot_idx],
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=12, fontweight="bold"
                    )
                    subplot_idx += 1

        # Legend and title
        fig.legend(["Pre", "Post"], loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2)
        fig.suptitle(f"{category.title()} CDF Pre vs Post by Condition (n={balance_info})", y=0.995)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        output_path = f"category_{category}_cdf_by_condition.png"
        plt.savefig(output_path, dpi=300)
        plt.close(fig)  

        print(f"Saved {output_path}")

import numpy as np
import matplotlib.pyplot as plt

def plot_category_proportion_bars_all(plot_data):
    """
    For each (focus, intervention, percentage) combination,
    plots pre/post bar charts for category composition within harmful content,
    and prints the values.
    """
    cdf_data = plot_data['cdf_data']
    categories = ['physical', 'hate', 'sexual']
    foci = plot_data['metadata']['foci']
    interventions = plot_data['metadata']['interventions']
    percentages = plot_data['metadata']['percentages']

    results = []

    for focus in foci:
        for intervention in interventions:
            for perc in percentages:
                # Collect pre/post category counts (as arrays)
                pre_vals = {}
                post_vals = {}
                total_pre = 0
                total_post = 0

                for cat in categories:
                    arr_pre = cdf_data.get(focus, {}).get(intervention, {}).get(perc, {}).get(cat, {}).get(1, np.array([]))
                    arr_post = cdf_data.get(focus, {}).get(intervention, {}).get(perc, {}).get(cat, {}).get(30, np.array([]))

                    # Use mean of arrays as the average proportion of harmful for this category
                    pre_val = float(np.mean(arr_pre)) if len(arr_pre) > 0 else 0
                    post_val = float(np.mean(arr_post)) if len(arr_post) > 0 else 0

                    pre_vals[cat] = pre_val
                    post_vals[cat] = post_val
                    total_pre += pre_val
                    total_post += post_val

                # Normalize so that sum to 1 (avoid divide by zero)
                if total_pre > 0:
                    pre_props = {cat: pre_vals[cat] / total_pre for cat in categories}
                else:
                    pre_props = {cat: 0 for cat in categories}
                if total_post > 0:
                    post_props = {cat: post_vals[cat] / total_post for cat in categories}
                else:
                    post_props = {cat: 0 for cat in categories}

                # Print out numerical values
                print(f"\nCondition: focus={focus}, intervention={intervention}, percentage={perc}%")
                print("Pre-intervention proportions:", pre_props)
                print("Post-intervention proportions:", post_props)

                # Store for later use (optional)
                results.append({
                    'focus': focus,
                    'intervention': intervention,
                    'percentage': perc,
                    'pre_props': pre_props,
                    'post_props': post_props
                })

                # Skip plotting if all data are zero
                if sum(pre_props.values()) == 0 and sum(post_props.values()) == 0:
                    continue

                # Plot
                x = np.arange(len(categories))
                width = 0.35

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(x - width/2, [pre_props[c] for c in categories], width, label='Pre')
                ax.bar(x + width/2, [post_props[c] for c in categories], width, label='Post')

                ax.set_ylabel('Proportion within Harmful')
                ax.set_title(f'{focus.title()}, {intervention.title()}, {perc}%')
                ax.set_xticks(x)
                ax.set_xticklabels([c.title() for c in categories])
                ax.legend()
                ax.set_ylim(0, 1)
                plt.tight_layout()
                output_path = f"category_bar.png"
                plt.savefig(output_path, dpi=300)
                plt.close()

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extract-data", action="store_true",
        help="Extract and save plot data arrays (run this first or when data changes)"
    )
    parser.add_argument(
        "--focus-filter", choices=["homepage", "upnext", "both"], default="both",
        help="Which focus group(s) to include: homepage, upnext, or both"
    )
    parser.add_argument(
        "--plot-type", choices=["cdf", "trends", "both", "in_harm_cdf", "in_harm_trends", "in_harm_both", "composition_bar", "composition_area", "composition_both", "original_cdf", "category_intervention"], default="cdf",
        help="Which plots to generate: cdf, trends, both, in_harm_cdf, in_harm_trends, in_harm_both, composition_bar, composition_area, composition_both, or original_cdf"
    )
    args = parser.parse_args()
    

    cache_path = "../.processing_cache.json"
    combined_dir = "../combined_puppets"
    data_path = "plot_data_arrays.pkl"

    if args.extract_data:
        # Extract and save plot data
        plot_data = extract_and_save_plot_data(cache_path, combined_dir, data_path)
    else:
        # Load pre-processed plot data
        plot_data = load_plot_data(data_path)
        
        # if True:
        #     cdf_data = plot_data['cdf_data']
        #     print(cdf_data.keys()) # 应该有 'homepage', 'upnext'
        #     print(cdf_data['homepage'].keys()) # ['none', 'replace', 'downrank']
        #     print(cdf_data['homepage']['replace'].keys()) # [0, 25, 50]
        #     print(cdf_data['homepage']['replace'][25].keys()) # ['harmful', 'sexual', 'hate', 'physical', ...]
        #     print(cdf_data['homepage']['replace'][25]['hate'].keys()) # [1, 30]
        #     print(cdf_data['homepage']['replace'][25]['hate'][30]) # numpy array
        #     break
    
        # Generate plots
        if args.plot_type in ["cdf", "both"]:
            # plot_category_cdf(plot_data, args.focus_filter)
            plot_all_category_cdf_by_condition(plot_data)
        
        if args.plot_type in ["trends", "both"]:
            plot_category_trends(plot_data, args.focus_filter)
        
        if args.plot_type in ["in_harm_cdf", "in_harm_both"]:
            plot_category_in_harm_cdf(plot_data, args.focus_filter)
        
        if args.plot_type in ["in_harm_trends", "in_harm_both"]:
            plot_category_in_harm_trends(plot_data, args.focus_filter)
        
        if args.plot_type in ["composition_bar", "composition_both"]:
            plot_category_proportion_bars_all(plot_data)
        
        if args.plot_type in ["composition_area", "composition_both"]:
            plot_category_composition_area(plot_data, args.focus_filter)
        
        if args.plot_type in ["original_cdf"]:
            plot_original_cdf_by_percentage(plot_data, args.focus_filter)
            
        if args.plot_type in ["category_intervention"]:
            ks_results = analyze_intervention_effects(plot_data, "both")

if __name__ == "__main__":
    main()