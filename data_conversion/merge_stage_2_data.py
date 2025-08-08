import pickle
import glob
import os
import numpy as np
from tqdm import tqdm

data_dir = "/p/ruishen/dllm_waymo_data/stage_2_data"

# Find all pkl files ending with 1000.pkl
pkl_files = glob.glob(os.path.join(data_dir, "*.pkl"))
for pkl_file in pkl_files:
    if "chunk" not in pkl_file:
        pkl_files.remove(pkl_file)

# Initialize merged data list
merged_data = []

# Load and merge all files
for pkl_file in tqdm(pkl_files, desc="Merging files"):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
        to_remove = []
        for key, value in data.items():
            if len(value) == 0:
                to_remove.append(key)
        for key in to_remove:
            del data[key]
        merged_data.extend(data)

print(f"Total merged data items: {len(merged_data)}")

# Save merged data
output_file = os.path.join(data_dir, "all_scenes_qa_data.pkl")
with open(output_file, 'wb') as f:
    pickle.dump(merged_data, f)

print(f"Merged data saved to {output_file}")