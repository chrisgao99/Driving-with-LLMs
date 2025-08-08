import os
import sys
import pickle
import numpy as np

from data_conversion.convert_data_utils import convert_to_descriptor_format
from data_conversion.convert_data import convert_data
from utils.new_prompt_utils import make_waymo_observation_prompt

def process_tfrecord(tfrecord_path, output_dir):
    tfrecord_name = tfrecord_path.split(".")[-1]
    try:
        vector_data = convert_data(tfrecord_path)
    except Exception as e:
        print(f"Failed to convert {tfrecord_path}: {e}")
        return

    generated_data = []

    for sid_egoid, scene_data in vector_data.items():
        converted_data = convert_to_descriptor_format(scene_data)
        frame_num = 0
        for data in converted_data:
            prompt = make_waymo_observation_prompt(data)
            frame_num += 1
            frame_data = {
                "frame_num": frame_num,
                "input": "",
                "langen": prompt,
                "observation": data
            }
            generated_data.append(frame_data)

    # if len(generated_data) > 100:
    #     generated_data = np.random.choice(generated_data, size=100, replace=False).tolist()

    # os.makedirs(output_dir, exist_ok=True)
    # output_path = os.path.join(output_dir, f"{tfrecord_name}.pkl")
    # with open(output_path, "wb") as f:
    #     pickle.dump(generated_data, f)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python generate_stage_1_data.py <batch_dir>")
        sys.exit(1)

    batch_dir = sys.argv[1]
    output_dir = "/p/ruishen/dllm_waymo_data/stage_1_data"

    for batch_file in os.listdir(batch_dir):
        if not batch_file.endswith(".txt"):
            continue
        batch_path = os.path.join(batch_dir, batch_file)
        with open(batch_path, "r") as f:
            for line in f:
                tfrecord_path = line.strip()
                if not tfrecord_path:
                    continue
                print("processing", tfrecord_path)
                process_tfrecord(tfrecord_path, output_dir)
