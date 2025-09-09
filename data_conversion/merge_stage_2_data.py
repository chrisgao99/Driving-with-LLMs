import pickle
import glob
import os
import numpy as np
from tqdm import tqdm

with open("/p/ruishen/dllm_waymo_data/stage_2_data/all_scenes_qa_data_chunk_1.pkl", 'rb') as f:
    data = pickle.load(f)

data_dir = "/p/ruishen/dllm_waymo_data/stage_2_data"

# Find all pkl files ending with 1000.pkl
pkl_files = glob.glob(os.path.join(data_dir, "*.pkl"))
pkl_files = list(pkl_files)
to_remove = []
for pkl_file in pkl_files:
    if "chunk" not in pkl_file:
        to_remove.append(pkl_file)
for pkl_file in to_remove:
    pkl_files.remove(pkl_file)

# Initialize merged data list
merged_data = {}

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
        merged_data.update(data)

print(f"Total merged data items: {len(merged_data)}")

# Save merged data
output_file = os.path.join(data_dir, "all_scenes_qa_data.pkl")
with open(output_file, 'wb') as f:
    pickle.dump(merged_data, f)

print(f"Merged data saved to {output_file}")

np.random.seed(42)

keys = list(merged_data.keys())

# sample training keys
sampled_keys = np.random.choice(keys, size=600, replace=False)

# remaining keys for test
remaining_keys = np.setdiff1d(keys, sampled_keys)
test_keys = np.random.choice(remaining_keys, size=60, replace=False)

# build train/test dicts
train_data = {k: merged_data[k] for k in sampled_keys}
test_data = {k: merged_data[k] for k in test_keys}

# Save train and test data
train_file = os.path.join(data_dir, "train_data.pkl")
test_file = os.path.join(data_dir, "test_data.pkl")
with open(train_file, 'wb') as f:
    pickle.dump(train_data, f)
with open(test_file, 'wb') as f:
    pickle.dump(test_data, f)