import os
import json

def main():
    puppets_dir = '../output/synced-puppets/puppets'
    total_files = 0
    success_count = 0
    failed_files = []

    for filename in os.listdir(puppets_dir):
        if not filename.endswith('.json'):
            continue

        filepath = os.path.join(puppets_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

        total_files += 1

        rounds = data.get('args', {}).get('rounds', None)
        actions = data.get('actions', [])

        if rounds is None or not isinstance(actions, list):
            failed_files.append(filename)
            continue

        # Collect all action names
        action_names = [a.get("action") for a in actions if "action" in a]

        has_current_round = f"round_{rounds}_end" in action_names
        has_next_round = f"round_{rounds + 1}_end" in action_names
        has_intervention_end = "intervention_end" in action_names

        if has_current_round and not has_next_round and has_intervention_end:
            success_count += 1
        else:
            failed_files.append(filename)

    print(f"Total puppet files checked: {total_files}")
    print(f"Successful puppets: {success_count}")
    print(f"Failed puppets: {total_files - success_count}")
    # if failed_files:
    #     print("Failed files:")
    #     for fname in failed_files:
    #         print(f" - {fname}")

if __name__ == '__main__':
    main()