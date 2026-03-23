import json
import pandas as pd

def analyze_user_stats(input_file):
    """
    Analyze watch histories to get statistics for each user.
    Returns user statistics and filtered users meeting criteria.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    user_stats = []
    qualified_users = []
    
    for user in data['users']:
        visa_id = user['visa_id']
        total_videos = len(user['videos'])
        
        # Count non-problematic unique videos
        non_problematic_videos = len([
            video for video in user['videos']
            if not video['is_problematic']
        ])
        
        problematic_videos = len([
            video for video in user['videos']
            if video['is_problematic']
        ])
        
        non_problematic_unique_videos = len(set([
            video['videoId'] for video in user['videos']
            if not video['is_problematic']
        ]))
        
        problematic_unique_videos = len(set([
            video['videoId'] for video in user['videos']
            if video['is_problematic']
        ]))
        
        stats = {
            'visa_id': visa_id,
            'total_unique_videos': total_videos,
            'non_problematic_videos': non_problematic_videos,
            'problematic_videos': problematic_videos,
            'non_problematic_unique_videos': non_problematic_unique_videos,
            'problematic_unique_videos': problematic_unique_videos
        }
        user_stats.append(stats)
        
        # Check if user meets criteria (50+ non-problematic videos)
        if non_problematic_videos >= 50:
            qualified_users.append(user)
    
    # Sort users by number of non-problematic videos
    user_stats.sort(key=lambda x: x['non_problematic_unique_videos'], reverse=True)
    
    return user_stats, qualified_users

def save_filtered_users(qualified_users, output_file, max_users=50):
    """
    Save up to 50 qualified users with their complete watch history to a new JSON file.
    """
    # Take first 50 users if we have more
    selected_users = qualified_users[:max_users]
    
    output_data = {
        "users": selected_users,
        "total_users": len(selected_users),
        "selection_criteria": {
            "description": "Users with 50+ non-problematic unique videos",
            "min_non_problematic_videos": 50
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

def print_stats_summary(user_stats):
    """
    Print summary statistics of user watch histories.
    """
    total_users = len(user_stats)
    qualified_users = len([u for u in user_stats if u['non_problematic_unique_videos'] >= 50])
    
    print(f"\nWatch History Statistics:")
    print(f"Total users analyzed: {total_users}")
    print(f"Users with 50+ non-problematic videos: {qualified_users}")
    print("\nTop 10 users by non-problematic videos:")
    print("VisaID | Total Videos | Non-problematic Videos")
    print("-" * 50)
    for stat in user_stats[:10]:
        print(f"{stat['visa_id']} | {stat['total_unique_videos']} | {stat['non_problematic_unique_videos']}")

def save_stats_summary(user_stats, summary_path):
    """
    Save summary statistics of user watch histories to csv file.
    """
    df = pd.DataFrame(user_stats)
    df.to_csv(summary_path, index=False)
    print(f"Saved user statistics to {summary_path}")
        
if __name__ == "__main__":
    input_file = "user_watch_histories_with_problematic.json"
    output_file = "filtered_users_50plus.json"
    
    # Analyze and get statistics
    user_stats, qualified_users = analyze_user_stats(input_file)
    
    # Print statistics
    print_stats_summary(user_stats)
    

    save_stats_summary(user_stats, "user_stats_summary.csv")
    
    # Save filtered users
    if len(qualified_users) >= 50:
        save_filtered_users(qualified_users, output_file)
        print(f"\nSaved {min(50, len(qualified_users))} qualified users to {output_file}")
    else:
        print(f"\nWarning: Only found {len(qualified_users)} users with 50+ non-problematic videos")