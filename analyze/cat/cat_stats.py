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
        
        for intervention in interventions:
            plot_data['cdf_data'][focus][intervention] = {}
            
            for category in all_categories:  # Now includes both ratio types
                plot_data['cdf_data'][focus][intervention][category] = {}
                
                # CDF data for rounds 1 and 30
                for round_num in [1, 30]:
                    values_by_perc = []
                    for perc in percentages:
                        key = (focus, intervention, perc, category, round_num)
                        if key in raw_data and len(raw_data[key]) >= min_size:
                            values_by_perc.extend(random.sample(raw_data[key], min_size))
                        elif key in raw_data:
                            values_by_perc.extend(raw_data[key])
                    
                    plot_data['cdf_data'][focus][intervention][category][round_num] = np.array(values_by_perc)
        
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
    Shows the proportion of each category within harmful content before and after intervention
    """
    print(f"=== PLOTTING CATEGORY COMPOSITION BAR CHART ({focus_filter}) ===")
    
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
    category_colors = dict(zip(categories_in_harm, colors))
    
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

            # Collect data for all categories
            pre_means = []
            post_means = []
            category_names = []
            
            for category_in_harm in categories_in_harm:
                category_name = category_in_harm.replace('_in_harm', '')
                
                if (focus in cdf_data and intervention in cdf_data[focus] and 
                    category_in_harm in cdf_data[focus][intervention]):
                    
                    r1 = cdf_data[focus][intervention][category_in_harm].get(1, np.array([]))
                    r30 = cdf_data[focus][intervention][category_in_harm].get(30, np.array([]))
                    
                    if len(r1) > 0 and len(r30) > 0:
                        pre_means.append(np.mean(r1))
                        post_means.append(np.mean(r30))
                        category_names.append(category_name.title())
                    else:
                        pre_means.append(0)
                        post_means.append(0)
                        category_names.append(category_name.title())
                else:
                    pre_means.append(0)
                    post_means.append(0)
                    category_names.append(category_in_harm.replace('_in_harm', '').title())

            # Create grouped bar chart
            x = np.arange(len(category_names))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, pre_means, width, label='Pre', alpha=0.8, color='lightblue')
            bars2 = ax.bar(x + width/2, post_means, width, label='Post', alpha=0.8, color='lightcoral')
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{height:.2f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),  # 3 points vertical offset
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=8)
            
            for bar in bars2:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{height:.2f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),  # 3 points vertical offset
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=8)

            ax.set_title(f"{focus} - {intervention}")
            ax.set_ylabel('Mean Proportion in Harmful Content')
            ax.set_xticks(x)
            ax.set_xticklabels(category_names)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f"Category Composition in Harmful Content: Pre vs Post (n={balance_info})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"category_composition_bar_{focus_filter}.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_category_composition_area(plot_data, focus_filter):
    """
    Plot stacked area chart showing category composition evolution within harmful content
    Shows how the proportion of each category within harmful content changes over time
    """
    print(f"=== PLOTTING CATEGORY COMPOSITION AREA CHART ({focus_filter}) ===")
    
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
    
    # Setup figure
    n_focus = len(foci)
    n_perc = len(percentages)
    
    fig, axes = plt.subplots(n_focus, n_perc, figsize=(5 * n_perc, 4 * n_focus), 
                            sharex=True, sharey=False)
    
    # Always ensure axes is 2D array
    if n_focus == 1 and n_perc == 1:
        axes = np.array([[axes]])
    elif n_focus == 1:
        axes = axes[np.newaxis, :]
    elif n_perc == 1:
        axes = axes[:, np.newaxis]

    # Color map for categories
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue
    
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
            
            # Create separate subplot for each intervention
            for k, intervention in enumerate(interventions):
                # Collect data for all categories in this condition
                category_data = {}
                
                for category_in_harm in categories_in_harm:
                    category_name = category_in_harm.replace('_in_harm', '')
                    
                    if (focus in trends_data and perc in trends_data[focus] and 
                        intervention in trends_data[focus][perc] and 
                        category_in_harm in trends_data[focus][perc][intervention]):
                        
                        data = trends_data[focus][perc][intervention][category_in_harm]
                        category_data[category_name] = data['mean']
                    else:
                        # Fill with zeros if no data
                        category_data[category_name] = np.zeros(max_rounds)
                
                # Prepare data for stacked area plot
                x = np.arange(1, max_rounds + 1)
                
                # Stack the areas
                bottom = np.zeros(max_rounds)
                for idx, (category_name, series) in enumerate(category_data.items()):
                    if k == 0:  # Only show legend for first intervention
                        ax.fill_between(x, bottom, bottom + series, 
                                       alpha=0.6, color=colors[idx], 
                                       label=f"{category_name.title()}")
                    else:
                        ax.fill_between(x, bottom, bottom + series, 
                                       alpha=0.6, color=colors[idx])
                    bottom += series
                
                # Add intervention lines to distinguish different strategies
                # Draw boundary lines between interventions (optional)
                if k < len(interventions) - 1:
                    ax.axvline(x=max_rounds//3 * (k+1), color='white', 
                              linestyle='--', alpha=0.7, linewidth=1)
            
            ax.set_title(f"{focus} - {perc}%")
            if i == n_focus - 1:
                ax.set_xlabel("Round")
            if j == 0:
                ax.set_ylabel("Proportion in Harmful Content")
            ax.set_ylim(0, 1)  # Proportions sum to 1
            ax.grid(True, alpha=0.3)

    # Set consistent y-axis range for each row (focus) independently
    for i in range(n_focus):
        row_ymin = 0  # Always start from 0 for proportions
        row_ymax = 1  # Maximum is 1 for proportions
        for j in range(n_perc):
            axes[i][j].set_ylim(row_ymin, row_ymax)

    # Add legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, 
                  loc='lower center', bbox_to_anchor=(0.5, 0), ncol=len(categories_in_harm))
    
    fig.suptitle(f"Category Composition Evolution in Harmful Content (n={balance_info})", fontsize=14)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(f"category_composition_area_{focus_filter}.png", dpi=300, bbox_inches='tight')
    plt.show()

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
        "--plot-type", choices=["cdf", "trends", "both", "in_harm_cdf", "in_harm_trends", "in_harm_both", "composition_bar", "composition_area", "composition_both"], default="cdf",
        help="Which plots to generate: cdf, trends, both, in_harm_cdf, in_harm_trends, in_harm_both, composition_bar, composition_area, or composition_both"
    )
    args = parser.parse_args()

    cache_path = ".processing_cache.json"
    combined_dir = "./combined_puppets"
    data_path = "plot_data_arrays.pkl"

    if args.extract_data:
        # Extract and save plot data
        plot_data = extract_and_save_plot_data(cache_path, combined_dir, data_path)
    else:
        # Load pre-processed plot data
        plot_data = load_plot_data(data_path)
        
        # Generate plots
        if args.plot_type in ["cdf", "both"]:
            plot_category_cdf(plot_data, args.focus_filter)
        
        if args.plot_type in ["trends", "both"]:
            plot_category_trends(plot_data, args.focus_filter)
        
        if args.plot_type in ["in_harm_cdf", "in_harm_both"]:
            plot_category_in_harm_cdf(plot_data, args.focus_filter)
        
        if args.plot_type in ["in_harm_trends", "in_harm_both"]:
            plot_category_in_harm_trends(plot_data, args.focus_filter)
        
        if args.plot_type in ["composition_bar", "composition_both"]:
            plot_category_composition_bar(plot_data, args.focus_filter)
        
        if args.plot_type in ["composition_area", "composition_both"]:
            plot_category_composition_area(plot_data, args.focus_filter)

if __name__ == "__main__":
    main()