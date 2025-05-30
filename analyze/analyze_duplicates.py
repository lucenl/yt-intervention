import json
import collections
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

# Input file from analyze_puppets.py
input_file = './success_puppets.json'

# Load puppet data
with open(input_file, 'r') as f:
    data = json.load(f)

# Part 1: Analyze duplicates in recommendations
rec_video_ids = []
for puppet in data:
    rec_video_ids.extend(puppet.get('homepage_recs', []))
    rec_video_ids.extend(puppet.get('upnext_recs', []))

print(f"Total number of video IDs collected from both homepage and upnext: {len(rec_video_ids)}")
rec_counter = collections.Counter(rec_video_ids)
total_rec_count = sum(rec_counter.values())
print(f"Total count of video IDs in recommendations: {total_rec_count}")

duplicates = {video_id: count for video_id, count in rec_counter.items() if count > 1}
print(f"Total duplicate video IDs in recommendations: {len(set(duplicates))}")

A_count = sum(1 for count in rec_counter.values() if count == 1)
B_count = sum(1 for count in rec_counter.values() if count > 1)
print(f"Number of videos that appear exactly once in recommendations (|A|): {A_count}")
print(f"Number of video IDs that appear more than once in recommendations (|B|): {B_count}")
print(f"Number of duplicate sum_|B| in recommendations (count(v)): {sum(count for count in rec_counter.values() if count > 1)}")

videos_A = [video_id for video_id, count in rec_counter.items() if count == 1]
videos_B = [video_id for video_id, count in rec_counter.items() if count > 1]
print(f"List of videos appearing exactly once in recommendations (|A|): {videos_A[:10]}...")
print(f"List of video IDs appearing more than once in recommendations (|B|): {videos_B[:10]}...")

# Plot recommendation frequency distribution
plt.figure(figsize=(18, 6))
n, bins, patches = plt.hist(rec_counter.values(), bins=range(1, max(rec_counter.values()) + 2), align='left', edgecolor='black')
plt.xlabel('Number of Occurrences')
plt.ylabel('Number of Video IDs')
for patch in patches:
    height = patch.get_height()
    if height > 0:
        plt.text(patch.get_x() + patch.get_width() / 2, height, int(height), ha='center', va='bottom')
plt.title('Frequency Distribution of Video IDs in Recommendations')
plt.xticks(range(1, max(rec_counter.values()) + 1))
plt.tight_layout()
plt.savefig('rec_frequency_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Part 2: Analyze training video occurrences
puppet_training = {}
training_counter = collections.Counter()
for puppet in data:
    puppet_id = puppet['puppet_id']
    # Check if training data is directly in the puppet or under args
    training_list = puppet.get('training', [])  # Try direct key first
    if not training_list:  # Fallback to args if not found
        training_list = puppet.get('args', {}).get('training', [])
    puppet_training[puppet_id] = training_list
    training_counter.update(training_list)

with open('training_videos.txt', 'w') as f:
    for video_id in training_counter.keys():
        f.write(f"{video_id}\n")

total_puppets = len(puppet_training)
unique_training_videos = len(training_counter)
print(f"\nTotal puppets analyzed: {total_puppets}")
print(f"Unique training videos across puppets: {unique_training_videos}")

training_video_puppet_map = {}
for puppet, training_list in puppet_training.items():
    for video in training_list:
        if video not in training_video_puppet_map:
            training_video_puppet_map[video] = set()
        training_video_puppet_map[video].add(puppet)

training_video_occurrences = {video: len(puppets) for video, puppets in training_video_puppet_map.items()}
duplicates_count = sum(1 for count in training_video_occurrences.values() if count > 1)
print(f"Number of training videos that appear in more than one puppet: {duplicates_count}")

all_occurrences = list(training_video_occurrences.values())
average_occurrence = sum(all_occurrences) / len(all_occurrences) if all_occurrences else 0
print(f"Average occurrence of each training video across puppets: {average_occurrence:.2f}")

top_training_videos = sorted(training_video_occurrences.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top 10 most common training videos among puppets:")
for video, count in top_training_videos:
    print(f"  {video}: in {count} puppets")

# Plot histogram of training video occurrences
plt.figure(figsize=(8, 6))
max_occurrence = max(training_video_occurrences.values()) if training_video_occurrences else 1
plt.hist(training_video_occurrences.values(), bins=range(1, max_occurrence + 2), align='left', edgecolor='black')
plt.xlabel('Number of Puppets a Training Video Appears In')
plt.ylabel('Number of Training Videos')
plt.title('Distribution of Training Video Occurrences Across Puppets')
plt.xticks(range(1, max_occurrence + 1))
plt.tight_layout()
plt.savefig('training_video_occurrences.png', dpi=300, bbox_inches='tight')
plt.close()

# Compute pairwise Jaccard similarity
def jaccard_similarity(set1, set2):
    if not set1 and not set2:
        return 1.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

jaccard_scores = []
puppet_ids = list(puppet_training.keys())
for p1, p2 in itertools.combinations(puppet_ids, 2):
    set1 = set(puppet_training[p1])
    set2 = set(puppet_training[p2])
    jaccard_scores.append(jaccard_similarity(set1, set2) if (set1 or set2) else 1.0)

if jaccard_scores:
    avg_jaccard = sum(jaccard_scores) / len(jaccard_scores)
    print(f"Average pairwise Jaccard similarity between puppet training sets: {avg_jaccard:.2f}")
else:
    print("Not enough puppets to compute pairwise Jaccard similarity.")

# Part 3: Violin plot for high-frequency video positions
freq_df = pd.DataFrame(list(rec_counter.items()), columns=['videoId', 'frequency']).sort_values('frequency', ascending=False)
freq_df.to_csv('video_frequencies.csv', index=False)
high_freq_videos = freq_df[freq_df['frequency'] >= 20]['videoId'].tolist()

position_data = []
for puppet in data:
    homepage_recs = puppet.get('homepage_recs', [])
    for video_id in high_freq_videos:
        if video_id in homepage_recs:
            position = homepage_recs.index(video_id)
            position_data.append({
                'video_id': video_id,
                'position': position,
                'harmful_percentage': puppet.get('harmful_percentage')
            })

pos_df = pd.DataFrame(position_data)
if not pos_df.empty:
    video_stats = pos_df.groupby('video_id')['position'].agg(['mean', 'min', 'max', 'count']).reset_index()
    video_stats = video_stats.sort_values('count', ascending=False)

    plt.figure(figsize=(16, 10))
    sns.set_style("whitegrid")

    # Use palette with hue to avoid FutureWarning
    ax = sns.violinplot(x='video_id', y='position', hue='video_id', data=pos_df, 
                       order=video_stats['video_id'], 
                       palette=sns.color_palette("Spectral", len(high_freq_videos))[:len(high_freq_videos)], 
                       inner='quartile', 
                       linewidth=1.5, 
                       saturation=0.85, 
                       width=0.9, 
                       legend=False)

    for i, row in enumerate(video_stats.itertuples()):
        plt.errorbar(i, row.mean, yerr=[[row.mean-row.min], [row.max-row.mean]],
                    fmt='o', 
                    markersize=12, 
                    color='white', 
                    markeredgecolor='black', 
                    markeredgewidth=2,
                    ecolor='#e41a1c', 
                    elinewidth=3, 
                    capsize=15, 
                    capthick=3)

    for i, row in enumerate(video_stats.itertuples()):
        plt.text(i, -1.5, f"n={row.count}", 
                ha='center', 
                fontsize=10,
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.gca().invert_yaxis()
    plt.ylim(pos_df['position'].max() + 2, -2)
    plt.xlabel('Video ID', fontsize=13, labelpad=15, fontweight='semibold')
    plt.ylabel('Position (0 = Top Recommendation)', fontsize=12, labelpad=15)
    plt.title('Distribution of High-Frequency Video Positions in Homepage Recommendations\n', 
             fontsize=16, fontweight='bold', pad=20)
    plt.xticks(range(len(video_stats)), 
              [f"{vid}\n(n={count})" for vid, count in zip(video_stats['video_id'], video_stats['count'])], 
              rotation=45, 
              ha='right',
              fontsize=11,
              fontfamily='monospace')
    plt.yticks(fontsize=11)
    plt.grid(axis='y', linestyle=':', alpha=0.4)

    legend_elements = [
        Patch(facecolor=sns.color_palette("Spectral", len(high_freq_videos))[0], label='Position Distribution'),
        plt.Line2D([0], [0], marker='o', color='w', label='Mean Position',
                  markerfacecolor='white', markeredgecolor='black', markersize=12),
        plt.Line2D([0], [0], color='#e41a1c', lw=3, label='Min/Max Range')
    ]
    plt.legend(handles=legend_elements, frameon=True, framealpha=0.9)

    plt.figtext(0.5, 0.01, 
        "Violin width shows position density | Error bars show full range | Mean marked with white circle",
        ha='center', fontsize=11, style='italic', color='#555555')
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    sns.despine(left=True)
    plt.savefig('position_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
else:
    print("No position data available for high-frequency videos.")