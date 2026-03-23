import os
import json

# Folder containing puppet JSON files.
puppet_dir = "../../output/puppets"
# File containing successful puppet IDs (one per line).
success_ids_file = "../../success_ids.txt"
# Folder where extracted argument files will be saved.
output_dir = "arguments"

# Ensure the output directory exists.
os.makedirs(output_dir, exist_ok=True)

# Read successful puppet IDs into a set.
with open(success_ids_file, 'r') as f:
    success_ids = {line.strip() for line in f if line.strip()}

print(f"Reading {len(success_ids)} successful puppet IDs from: {success_ids_file}")

# List all files in the puppet folder.
files = os.listdir(puppet_dir)
print(f"Found {len(files)} files in: {puppet_dir}")

# Process each puppet file.
for file in files:
    file_path = os.path.join(puppet_dir, file)
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        continue

    # Use "puppet_id" (not "puppetId") from the file.
    puppet_id = data.get("puppet_id")
    if not puppet_id:
        print(f"File {file} does not contain a 'puppet_id' field.")
        continue

    if puppet_id in success_ids:
        # Extract the "args" field.
        args = data.get("args")
        if args:
            args["outputDir"] = "/output"
            args["steps"] = "train"
            # Save the extracted args to a new file.
            out_file_path = os.path.join(output_dir, f"{puppet_id}.json")
            with open(out_file_path, 'w') as out_f:
                json.dump(args, out_f, indent=4)
            print(f"Extracted args for puppet {puppet_id} to {out_file_path}")
        else:
            print(f"No args found for puppet {puppet_id} in file {file_path}")
