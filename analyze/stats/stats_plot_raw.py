#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. Load the summary CSV
    df = pd.read_csv('puppet_experiment_summary_v4.csv')

    # 2. Reshape into long format
    round_cols = [f'num_harmful_videos_round_{i}' for i in range(100)]
    df_long = df.melt(
        id_vars=['puppet_id', 'intervention_type'],
        value_vars=round_cols,
        var_name='round',
        value_name='num_harmful'
    )
    df_long['round'] = (
        df_long['round']
        .str.replace('num_harmful_videos_round_', '')
        .astype(int)
    )

    # 3. Compute summary statistics by intervention_type and round
    grp = df_long.groupby(['intervention_type', 'round'])['num_harmful']
    summary = grp.agg(
        mean_harmful='mean',
        std_harmful='std',
        count='count'
    ).reset_index()
    summary['se_harmful'] = summary['std_harmful'] / np.sqrt(summary['count'])

    # 4. Plot the raw means with error bands (no smoothing)
    plt.figure(figsize=(10, 6))
    for strategy, sub in summary.groupby('intervention_type'):
        plt.plot(sub['round'], sub['mean_harmful'], label=strategy)
        lower = sub['mean_harmful'] - sub['se_harmful']
        upper = sub['mean_harmful'] + sub['se_harmful']
        plt.fill_between(sub['round'], lower, upper, alpha=0.2)

    plt.xlabel('Round')
    plt.ylabel('Average number of harmful videos')
    plt.title('Raw Harmful-Video Trends by Intervention Strategy')
    plt.legend(title='Intervention Type')
    plt.tight_layout()
    plt.savefig('harmful_trends_raw.png', dpi=300)
    plt.show()

    # 5. (Optional) Recompute slope and AUC on the raw means
    stats = []
    for strategy, sub in summary.groupby('intervention_type'):
        x = sub['round']
        y = sub['mean_harmful']
        slope = np.polyfit(x, y, 1)[0]
        auc = np.trapz(y, x)
        stats.append({
            'intervention_type': strategy,
            'slope_raw': slope,
            'auc_raw': auc
        })

    results_df = pd.DataFrame(stats)
    print('\nRaw‐mean summary (slope and AUC):\n')
    print(results_df.to_string(index=False))
    results_df.to_csv('strategy_summary_raw.csv', index=False)
    print('\nWrote raw‐mean summary to strategy_summary_raw.csv')

if __name__ == '__main__':
    main()
