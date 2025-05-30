#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. Load the summary CSV
    df = pd.read_csv('puppet_experiment_summary_v6.csv', dtype={'harmful_percent': str})
    df.columns = df.columns.str.strip()

    # 2. Reshape into long format
    round_cols = [f'num_harmful_videos_round_{i}' for i in range(100)]
    df_long = df.melt(
        id_vars=['puppet_id', 'intervention_type', 'harmful_percent'],
        value_vars=round_cols,
        var_name='round',
        value_name='num_harmful'
    )
    df_long['round'] = (
        df_long['round']
        .str.replace('num_harmful_videos_round_', '', regex=False)
        .astype(int)
    )

    # 3. Compute summary statistics by harmful_percent, intervention_type, and round
    grp = df_long.groupby(['harmful_percent', 'intervention_type', 'round'])['num_harmful']
    summary = grp.agg(
        mean_harmful='mean',
        std_harmful='std',
        count='count'
    ).reset_index()
    summary['se_harmful'] = summary['std_harmful'] / np.sqrt(summary['count'])

    # 4. Plot the raw means with error bands (no smoothing), one subplot per harmful_percent
    harmful_levels = sorted(summary['harmful_percent'].unique())
    fig, axes = plt.subplots(len(harmful_levels), 1, figsize=(12, 5 * len(harmful_levels)), sharex=True)

    if len(harmful_levels) == 1:
        axes = [axes]

    for ax, percent in zip(axes, harmful_levels):
        sub = summary[summary['harmful_percent'] == percent]
        for strategy in sub['intervention_type'].unique():
            curve = sub[sub['intervention_type'] == strategy]
            ax.plot(curve['round'], curve['mean_harmful'], label=strategy)
            lower = curve['mean_harmful'] - curve['se_harmful']
            upper = curve['mean_harmful'] + curve['se_harmful']
            ax.fill_between(curve['round'], lower, upper, alpha=0.2)
        ax.set_title(f'Harmful Percent: {percent}')
        ax.set_ylabel('Avg Harmful Videos')
        ax.legend(title='Strategy')
        ax.grid(True)

    axes[-1].set_xlabel('Round')
    plt.tight_layout()
    plt.savefig('harmful_trends_raw_v6.png', dpi=300)
    plt.show()

    # 5. Compute slope and AUC by both harmful_percent and strategy
    stats = []
    for (percent, strategy), sub in summary.groupby(['harmful_percent', 'intervention_type']):
        sub = sub.dropna(subset=['round', 'mean_harmful'])
        x = sub['round']
        y = sub['mean_harmful']
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            auc = np.trapz(y, x)
        else:
            slope = np.nan
            auc = np.nan
        stats.append({
            'harmful_percent': percent,
            'intervention_type': strategy,
            'slope_raw': slope,
            'auc_raw': auc
        })

    results_df = pd.DataFrame(stats)
    print('\nRaw‐mean summary (slope and AUC) per harmful_percent:\n')
    print(results_df.to_string(index=False))
    results_df.to_csv('strategy_summary_raw_by_percent.csv', index=False)
    print('\nWrote raw‐mean summary to strategy_summary_raw_by_percent.csv')

if __name__ == '__main__':
    main()
