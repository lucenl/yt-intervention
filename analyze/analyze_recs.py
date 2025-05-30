import json
import pandas as pd
import os
import csv

def extract_harmful_percentage(puppet):
    raw = puppet["harmful_percentage"]
    return int(raw) if raw is not None else None

def load_puppet_metadata(puppet_id, output_dir):
    """
    Load metadata from the puppet's metadata directory.
    """
    metadata_dir = os.path.join(output_dir, puppet_id, "metadata")
    if not os.path.exists(metadata_dir):
        print(f"Metadata directory {metadata_dir} not found.")
        return None
    
    homepage_df = None
    upnext_df = None
    for filename in os.listdir(metadata_dir):
        if filename.startswith("metadata_homepage_round_0.csv"):
            homepage_df = pd.read_csv(os.path.join(metadata_dir, filename), dtype=str, keep_default_na=False)
            homepage_df.set_index("video_id", inplace=True, drop=False)
        elif filename.startswith("metadata_upnext_round_0.csv"):
            upnext_df = pd.read_csv(os.path.join(metadata_dir, filename), dtype=str, keep_default_na=False)
            upnext_df.set_index("video_id", inplace=True, drop=False)
    
    return {"homepage": homepage_df, "upnext": upnext_df}

def expand_puppet(puppet, metadata_dict):
    """
    For one puppet dict, yield a row for each video in homepage/upnext with metadata.
    """
    percent = extract_harmful_percentage(puppet)
    pid = puppet["puppet_id"]
    output_dir = "output"  # Assuming output is the root directory

    for section, vids in [("homepage", puppet.get("homepage_recs", [])), ("upnext", puppet.get("upnext_recs", []))]:
        recs_df = metadata_dict.get(section)
        for vid in vids:
            if recs_df is not None and vid in recs_df.index:
                row = recs_df.loc[vid]
                yield {
                    "intended_harmful_percentage": percent,
                    "puppet_id": pid,
                    "section": section,
                    "video_id": vid,
                    "title": row.get("title", ""),
                    "description": row.get("description", ""),
                    "transcript": row.get("transcript", ""),
                    "prediction": row.get("prediction", None)
                }
            else:
                yield {
                    "intended_harmful_percentage": percent,
                    "puppet_id": pid,
                    "section": section,
                    "video_id": vid,
                    "title": "",
                    "description": "",
                    "transcript": "",
                    "prediction": None
                }

def process_file_pair(label, json_path):
    print(f"→ Processing {label}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    puppets = data if isinstance(data, list) else [data]

    rows = []
    for puppet in puppets:
        metadata_dict = load_puppet_metadata(puppet["puppet_id"], "output")
        rows.extend(expand_puppet(puppet, metadata_dict))

    out_df = pd.DataFrame(rows)
    out_csv = f"results_details_{label}.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"  saved → {out_csv}")
    return out_df

def main():
    config = [("all_puppets", "./success_puppets.json")]
    for label, json_path in config:
        process_file_pair(label, json_path)

if __name__ == "__main__":
    main()