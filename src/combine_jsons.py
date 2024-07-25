import os
import json

def combine_json_files(directory):
    combined_data = []

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            # Read the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)
                combined_data.append(data)

    # Define the output file path
    output_file_path = os.path.join(directory, 'combined.json')

    # Write the combined data to the output file
    with open(output_file_path, 'w') as output_file:
        json.dump(combined_data, output_file, indent=4)

    print(f"Combined JSON saved to {output_file_path}")

# Set the directory containing JSON files
json_directory = '/home/master_oogway/panda_ws/src/helping_hands_panda/data/annotations'  # Replace with the path to your directory

combine_json_files(json_directory)
