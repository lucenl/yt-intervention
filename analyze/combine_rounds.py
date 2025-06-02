import os
import json

def combine_puppet_jsons(puppet_txt_path, input_base_dir, output_dir, total_rounds=30):
    os.makedirs(output_dir, exist_ok=True)

    with open(puppet_txt_path, 'r') as f:
        puppet_ids = [line.strip() for line in f if line.strip()]

    for puppet_id in puppet_ids:
        puppet_data = {
            "puppet_id": puppet_id,
            "total_rounds": total_rounds,
            "rounds": {}
        }

        puppet_folder = os.path.join(input_base_dir, puppet_id)
        for round_num in range(1, total_rounds + 1):
            round_file = os.path.join(puppet_folder, f"round_{round_num}.json")
            if os.path.exists(round_file):
                with open(round_file, 'r') as f_json:
                    try:
                        round_data = json.load(f_json)
                        puppet_data["rounds"][f"round_{round_num}"] = round_data
                    except json.JSONDecodeError:
                        print(f"JSONDecodeError: Skipping {round_file}")
            else:
                print(f"Missing file: {round_file}")

        out_path = os.path.join(output_dir, f"{puppet_id}.json")
        with open(out_path, 'w') as f_out:
            json.dump(puppet_data, f_out, indent=2)

        print(f"Saved: {out_path}")

if __name__ == '__main__':
    combine_puppet_jsons(
        puppet_txt_path='experiment_stats_successful.txt',
        input_base_dir='../experiment_data',
        output_dir='./combined_puppets'
    )