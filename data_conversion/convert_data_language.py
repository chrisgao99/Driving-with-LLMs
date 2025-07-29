import numpy as np
from convert_data_utils import convert_to_descriptor_format
from get_data_for_eval import run_vectorize_process, find_nearby_road, find_nearby_agents
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from scripts.new_collect_vqa import get_qa_descriptor


def convert_data_language(tfrecord_path):
    data = run_vectorize_process(tfrecord_path, "language_condition")

    map_dict = data["map_dict"]
    tf_cleaned_traj_dict = data["tf_cleaned_traj_dict"]
    language_condition_data = data["language_condition_data"]
    valid_indices = data["valid_indices"]

    # for every one map in map_dict, find the corresponding ego trajectory and other agents' trajectories
    map_count = 0
    for sid_egoid, road_segments_list in map_dict.items():
        parts = sid_egoid.split("__")
        sid = parts[0]
        ego_id = int(parts[1])
        ego_traj = tf_cleaned_traj_dict[sid][ego_id]
        print(f"Processing scenario {sid} with ego ID {ego_id}")

        current_road, other_nearby_road = find_nearby_road(ego_traj['trajectory'], road_segments_list, proximity_threshold=5.0, n_road=6) # has keys 'type' and 'pos_xy'

        other_agent_trajs = {}

        other_agent_ids = valid_indices[sid][ego_id]['valid_agent']
        for agent_id in other_agent_ids:
            if agent_id in tf_cleaned_traj_dict[sid]:
                other_agent_trajs[agent_id] = tf_cleaned_traj_dict[sid][agent_id] 
        other_agent_trajs = find_nearby_agents(other_agent_trajs, ego_traj['trajectory'], n_agents=6)

        sample_data = {
            'Map Data': road_segments_list,
            'Ego Trajectory': ego_traj,
            'Nearby Agent Trajectories': other_agent_trajs,
            'Language Condition': language_condition_data[sid_egoid]
        }
        # print("language_condition_data: ", language_condition_data[sid_egoid])
        #convert the data from one map to a list of 19 dicts, one for each time step
        list_of_converted_data = convert_to_descriptor_format(sample_data)

        list_data_with_qa = get_qa_descriptor(list_of_converted_data, sample_data)


        map_count += 1
        print(f"Converted data for scenario {sid} with ego ID {ego_id} to descriptor format.")
        print(f"Number of time steps in converted data: {len(list_of_converted_data)}")     
        print(f"Number of time steps in converted data with QA: {len(list_data_with_qa)}")
        print(f"Converted {map_count} maps so far.")
        print("-----------------------------------------------------")



if __name__ == '__main__':
    filename = "/p/liverobotics/waymo_open_dataset_motion/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord-00000-of-00150"
    convert_data_language(filename)
