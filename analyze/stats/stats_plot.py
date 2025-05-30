#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 1. Load the summary CSV
    df = pd.read_csv('puppet_experiment_summary_v6.csv')

    # 2. Reshape into long format
    round_cols = [f'num_harmful_videos_round_{i}' for i in range(100)]
    df_long = df.melt(
        id_vars=['puppet_id', 'intervention_type', 'harmful_percent'],
        value_vars=round_cols,
        var_name='round',
        value_name='num_harmful'
    )
    df_long['round'] = df_long['round'] \
        .str.replace('num_harmful_videos_round_', '', regex=False) \
        .astype(int)

    # 3. Compute summary statistics by intervention_type and round
    grp = df_long.groupby(['harmful_percent', 'intervention_type', 'round'])['num_harmful']
    summary = grp.agg(
        mean_harmful='mean',
        std_harmful='std',
        count='count'
    ).reset_index()
    summary['se_harmful'] = summary['std_harmful'] / np.sqrt(summary['count'])

    # 4. Smooth the mean harmful counts with a rolling window (size = 5)
    summary['mean_smooth'] = summary.groupby(['harmful_percent', 'intervention_type'])['mean_harmful'] \
        .transform(lambda x: x.rolling(window=5, min_periods=1).mean())

    # 5. Plot the trends with error bands
    unique_percents = sorted(summary['harmful_percent'].unique())
    fig, axes = plt.subplots(len(unique_percents), 1, figsize=(12, 5 * len(unique_percents)), sharex=True)

    if len(unique_percents) == 1:
        axes = [axes]

    for ax, percent in zip(axes, unique_percents):
        sub = summary[summary['harmful_percent'] == percent]
        for strategy in sub['intervention_type'].unique():
            curve = sub[sub['intervention_type'] == strategy]
            ax.plot(curve['round'], curve['mean_smooth'], label=strategy)
            lower = curve['mean_smooth'] - curve['se_harmful']
            upper = curve['mean_smooth'] + curve['se_harmful']
            ax.fill_between(curve['round'], lower, upper, alpha=0.2)
        ax.set_title(f'{percent} Harmful Percent')
        ax.set_ylabel('Avg Harmful Videos')
        ax.legend(title='Intervention Type')

    axes[-1].set_xlabel('Round')
    plt.tight_layout()
    plt.savefig('puppet_harmful_trends_v6.png', dpi=300)
    plt.show()

    # 6. Compute slope and AUC for each strategy (across all harmful_percent values)
    stats = []
    for (strategy), sub in summary.groupby('intervention_type'):
        x = sub['round']
        y = sub['mean_smooth']
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            auc = np.trapz(y, x)
        else:
            slope = np.nan
            auc = np.nan
        stats.append({
            'intervention_type': strategy,
            'slope': slope,
            'auc': auc
        })

    results_df = pd.DataFrame(stats)
    print('\nSummary of slopes and AUCs by strategy:\n')
    print(results_df.to_string(index=False))
    results_df.to_csv('strategy_summary.csv', index=False)
    print('\nWrote detailed summary to strategy_summary.csv')

if __name__ == '__main__':
    main()
