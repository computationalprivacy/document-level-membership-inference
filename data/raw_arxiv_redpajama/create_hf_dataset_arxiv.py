import json
import os
from datasets import Dataset
from tqdm import tqdm 

# Directory containing your jsonl files
files_directory = "XX"
path_to_result = "XX"

# List to store dataset entries
dataset_entries = []

# Loop through each file in the directory
for filename in tqdm(os.listdir(files_directory)):
    if filename.endswith(".jsonl"):
        # let's extract the entries in the jsonl file
        with open(filename, 'r') as json_file:
            json_list = list(json_file)

        # let's now add the data
        for json_str in tqdm(json_list):
            try:
                paper = json.loads(json_str)
                dataset_entries.append(paper)
            except Exception as e:
                print(e)

print('Number of arxiv papers: ', len(dataset_entries))

# Create the dataset
dataset = Dataset.from_dict({"meta": [entry["meta"] for entry in dataset_entries],
                             "text": [entry["text"] for entry in dataset_entries]})

# Save the dataset
dataset.save_to_disk(path_to_result)