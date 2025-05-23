import os
import json
import csv

# Paths to puppet definitions and experiment results
puppet_dir = '/media/data/lucen/codebase/yt-sock-puppet/output_v4_success/puppets'
experiment_dir = '/media/data/lucen/codebase/yt-sock-puppet/experiment_data_v4_success'
output_csv = 'puppet_experiment_summary_v4.csv'

# Find all puppet JSON files
puppet_files = [
    os.path.join(puppet_dir, fname)
    for fname in os.listdir(puppet_dir)
    if fname.endswith('.json')
]
print(f'Found {len(puppet_files)} puppet files.')


rows = []

for puppet_path in puppet_files:
    # Load puppet definition
    with open(puppet_path, 'r') as f:
        puppet = json.load(f)

    # Extract puppet parameters
    puppet_id = puppet.get('puppet_id')
    args = puppet.get('args', {})
    trainingN = args.get('trainingN')
    steps = args.get('steps')
    rounds = args.get('rounds')
    focus = args.get('focus')
    intervention_type = args.get('intervention_type')

    # Prepare base data for this puppet
    data = {
        'puppet_id': puppet_id,
        'trainingN': trainingN,
        'steps': steps,
        'rounds': rounds,
        'focus': focus,
        'intervention_type': intervention_type
    }

    # Determine how many rounds to load
    max_rounds = rounds
    result_dir = os.path.join(experiment_dir, str(puppet_id))

    # Load each round result
    for i in range(max_rounds):
        result_file = os.path.join(result_dir, f'round_{i}.json')
        try:
            with open(result_file, 'r') as rf:
                result = json.load(rf)
        except FileNotFoundError:
            print(f'Warning: Result file {result_file} not found. Skipping round {i}.')
            data['rounds'] = i - 1
            continue
        except Exception as e:
            print(f'Unexpected error loading {result_file}: {e}')
            continue

        # Extract metrics from result
        num_harm = result.get('num_harmful_videos')
        harm_counts = result.get('harm_category_counts', {})

        # Add them to the row, flattening the harm counts as a JSON string
        data[f'num_harmful_videos_round_{i}'] = num_harm
        data[f'harm_category_counts_round_{i}'] = json.dumps(harm_counts)

    rows.append(data)
    
    
base_fields = [
    'puppet_id',
    'trainingN',
    'steps',
    'rounds',
    'focus',
    'intervention_type'
]

# then 100 rounds of metrics, in order
round_fields = []
for i in range(100):
    round_fields.append(f'num_harmful_videos_round_{i}')
    round_fields.append(f'harm_category_counts_round_{i}')

# final fieldnames list
fieldnames = base_fields + round_fields

# Write summary CSV
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f'Summary written to {output_csv}')
