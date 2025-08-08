import pickle
import glob
import os
import numpy as np
from tqdm import tqdm

data_dir = "/p/liverobotics/Rui/Driving-with-LLMs/stage_1_data"

# Find all pkl files ending with 1000.pkl
pkl_files = glob.glob(os.path.join(data_dir, "*1000.pkl"))
print(f"Found {len(pkl_files)} files to merge")

# Initialize merged data list
merged_data = []

average_length = int(100000 / len(pkl_files))

# Load and merge all files
for pkl_file in tqdm(pkl_files, desc="Merging files"):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
        if len(data) > average_length:
            data = np.random.choice(data, size=average_length, replace=False).tolist()
        merged_data.extend(data)

print(f"Total merged data items: {len(merged_data)}")

# Save merged data
output_file = os.path.join(data_dir, "merged_100k_data.pkl")
with open(output_file, 'wb') as f:
    pickle.dump(merged_data, f)

print(f"Merged data saved to {output_file}")