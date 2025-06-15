import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import random
import json
from pathlib import Path

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


def generate_comprehensive_dataset(cache, combined, output_path="comprehensive_puppet_data.csv"):
    """
    Generate a comprehensive dataset with all puppet data for further analysis
    Each row represents one puppet with all round-by-round data
    """
    print("Generating comprehensive dataset...")
    
    data_rows = []
    
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
            
        # Base row with puppet metadata
        row = {
            'puppet_id': puppet_id,
            'focus': focus,
            'harmful_percentage': perc,
            'intervention_type': intervention
        }
        
        # Add round-by-round data
        rounds_data = puppet_json.get("rounds", {})
        for round_num in range(1, 31):  # max_rounds = 30
            round_key = f"round_{round_num}"
            round_data = rounds_data.get(round_key, {})
            
            # Number of harmful videos this round
            harmful_count = round_data.get("num_harmful_videos", 0)
            recommendations = round_data.get("recommendations", [])
            total_recs = len(recommendations)
            
            # Calculate harmful ratio for this round
            harmful_ratio = harmful_count / total_recs if total_recs > 0 else 0
            
            # Add to row
            row[f'round_{round_num}_harmful_count'] = harmful_count
            row[f'round_{round_num}_total_recs'] = total_recs
            row[f'round_{round_num}_harmful_ratio'] = harmful_ratio
        
        # Add cumulative totals from the puppet JSON
        total_data = puppet_json.get("total", {})
        row['total_harmful_videos'] = total_data.get("num_harmful_videos", 0)
        row['total_recommendations'] = total_data.get("total_recommendations", 0)
        row['overall_harmful_ratio'] = (row['total_harmful_videos'] / row['total_recommendations'] 
                                      if row['total_recommendations'] > 0 else 0)
        
        data_rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data_rows)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Comprehensive dataset saved to {output_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df

def generate_descriptive_stats_table(df, balanced=True, output_path="descriptive_stats.csv"):
    """
    Generate Table A1: Descriptive Statistics for Harmful Video Exposure
    """
    print(f"Generating descriptive statistics table (balanced={balanced})...")
    
    # Focus on Round 1 and Round 30 harmful ratios
    results = []
    
    # Group by conditions
    conditions = df.groupby(['focus', 'intervention_type', 'harmful_percentage'])
    
    if balanced:
        # Find minimum sample size for balancing
        min_size = min([len(group) for _, group in conditions])
        print(f"Balancing to minimum sample size: {min_size}")
        
        # Set random seed for reproducibility
        random.seed(42)
    
    for (focus, intervention, harm_perc), group in conditions:
        if balanced and len(group) > min_size:
            # Sample to balance
            group = group.sample(n=min_size, random_state=42)
        
        n = len(group)
        
        # Round 1 stats
        round1_data = group['round_1_harmful_ratio']
        round1_mean = round1_data.mean()
        round1_std = round1_data.std()
        round1_median = round1_data.median()
        round1_q25 = round1_data.quantile(0.25)
        round1_q75 = round1_data.quantile(0.75)
        
        # Round 30 stats
        round30_data = group['round_30_harmful_ratio']
        round30_mean = round30_data.mean()
        round30_std = round30_data.std()
        round30_median = round30_data.median()
        round30_q25 = round30_data.quantile(0.25)
        round30_q75 = round30_data.quantile(0.75)
        
        results.append({
            'Context': focus,
            'Strategy': intervention,
            'Training_Harm_Percent': f"{harm_perc}%",
            'Round_1_Mean': f"{round1_mean:.3f}",
            'Round_1_SD': f"{round1_std:.3f}",
            'Round_1_Median': f"{round1_median:.3f}",
            'Round_1_IQR': f"{round1_q25:.3f}-{round1_q75:.3f}",
            'Round_30_Mean': f"{round30_mean:.3f}",
            'Round_30_SD': f"{round30_std:.3f}",
            'Round_30_Median': f"{round30_median:.3f}",
            'Round_30_IQR': f"{round30_q25:.3f}-{round30_q75:.3f}",
            'N': n
        })
    
    # Create DataFrame and save
    stats_df = pd.DataFrame(results)
    stats_df = stats_df.sort_values(['Context', 'Strategy', 'Training_Harm_Percent'])
    
    balanced_suffix = "_balanced" if balanced else "_unbalanced"
    final_output_path = output_path.replace(".csv", f"{balanced_suffix}.csv")
    stats_df.to_csv(final_output_path, index=False)
    
    print(f"Descriptive statistics table saved to {final_output_path}")
    return stats_df

def generate_ks_test_table(df, balanced=True, output_path="ks_test_results.csv"):
    """
    Generate Table A2: Complete Kolmogorov-Smirnov Test Results
    """
    print(f"Generating KS test results table (balanced={balanced})...")
    
    results = []
    
    # Group by conditions
    conditions = df.groupby(['focus', 'intervention_type', 'harmful_percentage'])
    
    if balanced:
        # Find minimum sample size for balancing
        min_size = min([len(group) for _, group in conditions])
        print(f"Balancing to minimum sample size: {min_size}")
        
        # Set random seed for reproducibility
        random.seed(42)
    
    for (focus, intervention, harm_perc), group in conditions:
        if balanced and len(group) > min_size:
            # Sample to balance
            group = group.sample(n=min_size, random_state=42)
        
        # Get Round 1 and Round 30 data
        round1_data = group['round_1_harmful_ratio'].values
        round30_data = group['round_30_harmful_ratio'].values
        
        # Perform KS test
        if len(round1_data) > 0 and len(round30_data) > 0:
            ks_stat, p_value = ks_2samp(round1_data, round30_data)
            
            # Effect size interpretation
            if ks_stat < 0.1:
                effect_size = "Negligible"
            elif ks_stat < 0.3:
                effect_size = "Small"
            elif ks_stat < 0.5:
                effect_size = "Medium"
            else:
                effect_size = "Large"
            
            # Significance markers
            if p_value < 0.001:
                significance = "***"
            elif p_value < 0.01:
                significance = "**"
            elif p_value < 0.05:
                significance = "*"
            else:
                significance = ""
            
            results.append({
                'Context': focus,
                'Strategy': intervention,
                'Training_Harm_Percent': f"{harm_perc}%",
                'KS_Statistic': f"{ks_stat:.3f}{significance}",
                'p_value': f"{p_value:.6f}",
                'Effect_Size': effect_size,
                'N': len(group)
            })
    
    # Create DataFrame and save
    ks_df = pd.DataFrame(results)
    ks_df = ks_df.sort_values(['Context', 'Strategy', 'Training_Harm_Percent'])
    
    balanced_suffix = "_balanced" if balanced else "_unbalanced"
    final_output_path = output_path.replace(".csv", f"{balanced_suffix}.csv")
    ks_df.to_csv(final_output_path, index=False)
    
    print(f"KS test results table saved to {final_output_path}")
    return ks_df

def generate_all_tables(cache_path, combined_dir):
    """
    Main function to generate all tables from the data
    """
    print("Loading data...")
    cache = load_cache(cache_path)
    combined = load_combined_puppets(combined_dir)
    
    # Generate comprehensive dataset
    df = generate_comprehensive_dataset(cache, combined)
    
    # Generate both balanced and unbalanced versions of tables
    print("\n" + "="*50)
    print("Generating unbalanced tables...")
    desc_unbalanced = generate_descriptive_stats_table(df, balanced=False)
    ks_unbalanced = generate_ks_test_table(df, balanced=False)
    
    print("\n" + "="*50)
    print("Generating balanced tables...")
    desc_balanced = generate_descriptive_stats_table(df, balanced=True)
    ks_balanced = generate_ks_test_table(df, balanced=True)
    
    print("\n" + "="*50)
    print("All tables generated successfully!")
    
    return {
        'comprehensive_data': df,
        'descriptive_unbalanced': desc_unbalanced,
        'descriptive_balanced': desc_balanced,
        'ks_unbalanced': ks_unbalanced,
        'ks_balanced': ks_balanced
    }

# Example usage:
if __name__ == "__main__":
    # You'll need to import the load functions from your original script
    # from your_original_script import load_cache, load_combined_puppets
    
    cache_path = ".processing_cache.json"
    combined_dir = "./combined_puppets"
    
    tables = generate_all_tables(cache_path, combined_dir)
    
    # Display sample of the comprehensive data
    print("\nSample of comprehensive dataset:")
    print(tables['comprehensive_data'].head())
    
    print("\nDescriptive statistics (balanced):")
    print(tables['descriptive_balanced'].head())
    
    print("\nKS test results (balanced):")
    print(tables['ks_balanced'].head())