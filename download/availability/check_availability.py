import requests
import json
import csv
import os
from dotenv import load_dotenv

# Load API key from environment variables
load_dotenv()
API_KEY = "AIzaSyCGDOfQcocrRlzIC1ixUt60iluuEC3wBMk"
BATCH_SIZE = 50  # Maximum allowed video IDs per API call


def check_video_restrictions_batch(video_ids):
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "contentDetails,status",
        "id": ",".join(video_ids),
        "key": API_KEY
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        # Return error for each video in the batch if the request fails.
        return {vid: {"error": f"Error: Unable to fetch data ({response.status_code})"} for vid in video_ids}

    data = response.json()
    results = {}

    # Process each returned item.
    if "items" in data:
        for item in data["items"]:

            vid = item["id"]
            restrictions = {
                "requires_sign_in": False,
                "private": item["status"]["privacyStatus"],  # Changed from privacy_status to private
                "age_restricted": False,
                "region_restricted": False,
                "region_restricted_in_US": False
            }

            # Check if video is private.
            if item["status"]["privacyStatus"] == "private":
                restrictions["requires_sign_in"] = True

            # Check for age restrictions.
            if ("contentRating" in item["contentDetails"] and
                    "ytRating" in item["contentDetails"]["contentRating"]):
                if item["contentDetails"]["contentRating"]["ytRating"] == "ytAgeRestricted":
                    restrictions["requires_sign_in"] = True
                    restrictions["age_restricted"] = True

            # Check for region restrictions.
            if "regionRestriction" in item["contentDetails"]:
                restrictions["region_restricted"] = True
                if "blocked" in item["contentDetails"]["regionRestriction"]:
                    if "US" in item["contentDetails"]["regionRestriction"]["blocked"]:
                        restrictions["region_restricted_in_US"] = True

            results[vid] = restrictions

    # For video IDs not returned, mark them as not found.
    for vid in video_ids:
        if vid not in results:
            results[vid] = {"error": "Error: Video not found or removed."}

    return results


def process_videos(csv_file, json_file):
    try:
        # Read video IDs from CSV.
        video_ids = []
        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Check for both possible column names
                vid = row.get("videoId")
                if vid:
                    video_ids.append(vid)

        if not video_ids:
            print(f"No video IDs found in {csv_file}. Check column names.")
            return

        results = []
        unwatchable_count = 0
        unwatchable_details = {
            "requires_sign_in": 0,
            "private": 0,
            "age_restricted": 0,
            "region_restricted": 0,
            "region_restricted_in_US": 0,
            "error": 0
        }

        # Process videos in batches.
        for i in range(0, len(video_ids), BATCH_SIZE):
            batch = video_ids[i:i+BATCH_SIZE]
            batch_results = check_video_restrictions_batch(batch)

            for vid in batch:
                result = batch_results[vid]

                is_unwatchable = False
                # Build a list of reasons for unwatchability.
                reason = []
                if "error" in result:
                    reason.append(result["error"])
                else:
                    for key in ["private", "age_restricted", "region_restricted_in_US"]:
                        if key == "private":
                            if result.get(key) == "private":
                                reason.append(key)
                        else:
                            if result.get(key):
                                reason.append(key)

                video_result = {"video_id": vid, "restrictions": result, "reason": reason}
                results.append(video_result)

                if "error" in result:
                    is_unwatchable = True
                    unwatchable_details["error"] += 1
                else:
                    if result.get("requires_sign_in"):
                        unwatchable_details["requires_sign_in"] += 1
                        is_unwatchable = True
                    if result.get("private") == 'private':  # Changed from privacy_status to private
                        unwatchable_details["private"] += 1
                        is_unwatchable = True
                    if result.get("age_restricted"):
                        is_unwatchable = True
                        unwatchable_details["age_restricted"] += 1
                    if result.get("region_restricted"):
                        is_unwatchable = True
                        unwatchable_details["region_restricted"] += 1
                        if result.get("region_restricted_in_US"):
                            unwatchable_details["region_restricted_in_US"] += 1

                if is_unwatchable:
                    unwatchable_count += 1

        output = {
            "results": results,
            "unwatchable_count": unwatchable_count,
            "unwatchable_details": unwatchable_details
        }

        with open(json_file, "w") as f:
            json.dump(output, f, indent=4)

        print("Processed", len(results), "videos.")
        print("Unwatchable videos:", unwatchable_count)
        print("Details:", unwatchable_details)

    except FileNotFoundError:
        print(f"Error: File not found - {csv_file}")
    except Exception as e:
        print(f"Error processing videos: {str(e)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check YouTube video restrictions")
    parser.add_argument("--csv", default="harmful_cleaned.csv", help="CSV file with video IDs")
    parser.add_argument("--output", default="cleaned_results.json", help="Output JSON file")

    args = parser.parse_args()

    process_videos(args.csv, args.output)
