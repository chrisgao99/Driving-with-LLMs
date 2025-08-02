import numpy as np
from data_conversion.convert_data_utils import convert_to_descriptor_format
from data_conversion.get_data_for_eval import run_vectorize_process, find_nearby_road, find_nearby_agents


def convert_data(tfrecord_path):
    """
    Converts the Waymo dataset from TFRecord format to a structured format.
    
    Args:
        tfrecord_name (str): The name of the TFRecord file to convert.
        setting (str): The setting for conversion, default is "default".
        
    Returns:
        dict: A dictionary containing the converted data.
    """
    data = run_vectorize_process(tfrecord_path, "language_condition")

    map_dict = data["map_dict"]
    tf_cleaned_traj_dict = data["tf_cleaned_traj_dict"]
    language_condition_data = data["language_condition_data"]
    valid_indices = data["valid_indices"]

    output_data = {}

    # for every one map in map_dict, find the corresponding ego trajectory and other agents' trajectories
    map_count = 0
    for sid_egoid, road_segments_list in map_dict.items():
        parts = sid_egoid.split("__")
        sid = parts[0]
        ego_id = int(parts[1])
        ego_traj = tf_cleaned_traj_dict[sid][ego_id]

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
            'Nearby Agent Trajectories': other_agent_trajs
        }
        
        output_data[sid_egoid] = sample_data
    
    return output_data


if __name__ == '__main__':
    filename = "/p/liverobotics/waymo_open_dataset_motion/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord-00000-of-00150"
    convert_data(filename)
