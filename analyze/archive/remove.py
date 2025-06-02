import os
import subprocess

# Configuration
PROFILES_DIR = "/media/data/lucen/codebase/yt-sock-puppet/output/profiles"
PROFILES_TO_KEEP = {
    "harmful_0,c55f31b4",
    "harmful_0,cb6b1a71",
    "harmful_5,cefdcc79",
    "harmful_5,e38b761b",
    "harmful_30,8d0aa832",
    "harmful_30,e780550d6",
    "harmful_50,46bfa92b",
    "harmful_50,36221166",
    "harmful_70,2fee18b0",
    "harmful_70,54f33058"
}

def get_profiles():
    """List all profile directories."""
    profiles = [d for d in os.listdir(PROFILES_DIR) if os.path.isdir(os.path.join(PROFILES_DIR, d))]
    return profiles

def delete_profile(profile_path):
    """Delete a profile directory using sudo."""
    try:
        subprocess.run(["sudo", "rm", "-rf", profile_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Deleted: {profile_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error deleting {profile_path}: {e.stderr.decode()}")

def keep_specific_profiles():
    """Keep the specified 10 profiles, delete the rest using sudo."""
    profiles = get_profiles()
    total_profiles = len(profiles)
    print(f"Found {total_profiles} profiles.")

    profiles_to_delete = [p for p in profiles if p not in PROFILES_TO_KEEP]

    if not profiles_to_delete:
        print("No profiles to delete. All existing profiles are in the keep list.")
        return

    print(f"Keeping {len(PROFILES_TO_KEEP)} profiles:")
    for profile in PROFILES_TO_KEEP:
        print(f"  - {profile}")

    print(f"\nDeleting {len(profiles_to_delete)} profiles:")
    for profile in profiles_to_delete:
        profile_path = os.path.join(PROFILES_DIR, profile)
        delete_profile(profile_path)

    print(f"\nKept {len(PROFILES_TO_KEEP)} profiles, deleted {len(profiles_to_delete)} profiles.")

if __name__ == "__main__":
    keep_specific_profiles()