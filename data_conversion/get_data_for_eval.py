import json
import traceback
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from get_data_utils import process_map_data, extract_tfrecord, format_dict, extract_road_segments


def process_traj_data(trajectories_dict, ego_agent_id, proximity_radius=5.0):
    """
    Filters agents based on their trajectory's proximity to an ego agent's trajectory.

    Args:
        trajectories_dict (dict): A dictionary where keys are agent IDs and values
                                  are their trajectories (a list of [x, y] points).
        ego_agent_id (str): The ID of the ego agent.
        proximity_radius (float): The maximum distance an agent's trajectory can be
                                  from the ego's trajectory to be included.

    Returns:
        dict: A new dictionary containing only the agents (including the ego)
              that are within the proximity radius.
    """
    if ego_agent_id not in trajectories_dict:
        return []

    ego_trajectory = trajectories_dict[ego_agent_id]
    ego_trajectory = ego_trajectory['trajectory']
    ego_trajectory = [point for point in ego_trajectory if point[0] != -1 and point[1] != -1]
    
    # Initialize the new dictionary with the ego agent
    filtered_agents = [ego_agent_id]

    # Check every other agent
    for agent_id, agent_trajectory in trajectories_dict.items():
        if agent_id == ego_agent_id:
            continue

        agent_trajectory = agent_trajectory['trajectory']

        # Check if any point of the agent's trajectory is close to the ego's path
        for point in agent_trajectory:
            dist = np.linalg.norm(np.array(ego_trajectory) - np.array(point), axis=1)
            if np.any(dist <= proximity_radius):
                # If one point is close enough, add the agent and stop checking this agent
                filtered_agents.append(agent_id)
                break
                
    return filtered_agents

def process_trajectory_dict(traj_dict):
    result = {}
    for sid, agent_traj in traj_dict.items():
        result[sid] = {}
        for agent_id, type_traj in agent_traj.items():
            agent_traj = type_traj['trajectory']
            agent_type = type_traj['type']
            valid_states = sum(1 for point in agent_traj if point[0] != -1 and point[1] != -1)
            if valid_states < 20:
                continue
            result[sid][agent_id] = {'type': agent_type, 'trajectory': agent_traj}
    return result

def find_nearby_road(ego_trajectory, road_segments, proximity_threshold=5.0, n_road=6):
    """
    Identifies the road segment an ego vehicle is currently on and finds other nearby roads.

    Args:
        ego_trajectory (list): A list of [x, y] points for the ego vehicle.
        road_segments (list): A list of road segment dictionaries from the map.
        proximity_threshold (float): The maximum distance (in meters) to be
                                     considered "on" a road. A standard lane
                                     is ~3.7m wide, so half of that is a good start.
        n_road (int): The number of nearby roads to output.

    Returns:
        current_road (dict): The road segment the ego vehicle is currently on.
        other_nearby_road (list): A list of other nearby road segments ranked by distance.
    """
    if len(road_segments) == 0 or len(ego_trajectory) == 0:
        return None, []
    
    # Calculate distances for all road segments
    road_distances = []
    
    for i, segment in enumerate(road_segments):
        if not segment['pos_xy']:
            continue
            
        # Calculate minimum distance from ego trajectory to this road segment
        dist = np.linalg.norm(np.array(segment['pos_xy'])[:, None, :] - np.array(ego_trajectory)[None, :, :], axis=2)
        min_dist = dist.min()
        
        road_distances.append((min_dist, segment))
    
    # Sort by distance
    road_distances.sort(key=lambda x: x[0])
    
    # Find current road (closest within proximity threshold)
    current_road = None
    other_nearby_road = []
    
    for dist, segment in road_distances[:n_road]:
        if current_road is None and dist <= proximity_threshold:
            current_road = segment
        else:
            other_nearby_road.append(segment)
    
    # If no road is within proximity threshold, use the closest one as current
    if current_road is None and road_distances:
        current_road = road_distances[0][1]
        other_nearby_road = [segment for _, segment in road_distances[1:n_road]]
    
    return current_road, other_nearby_road

def find_nearby_agents(other_agent_trajs, ego_trajectory, n_agents=6):
    """
    Filters and ranks other agents by their minimum distance to the ego trajectory.

    Args:
        other_agent_trajs (dict): A dictionary where keys are agent IDs and values
                                  are agent data with 'type' and 'trajectory' keys.
        ego_trajectory (list): A list of [x, y] points for the ego vehicle.
        n_agents (int): The maximum number of nearby agents to return.

    Returns:
        dict: A filtered dictionary containing the nearest agents ranked by distance.
    """
    if len(other_agent_trajs) == 0 or len(ego_trajectory) == 0:
        return {}
    
    # Calculate distances for all agents
    agent_distances = []
    ego_traj_array = np.array(ego_trajectory)
    
    for agent_id, agent_data in other_agent_trajs.items():
        agent_trajectory = agent_data['trajectory']
        # Filter out invalid points
        valid_points = [point for point in agent_trajectory if point[0] != -1 and point[1] != -1]
        
        if not valid_points:
            continue
            
        # Calculate minimum distance from agent trajectory to ego trajectory
        agent_traj_array = np.array(valid_points)
        dist = np.linalg.norm(ego_traj_array[:, None, :] - agent_traj_array[None, :, :], axis=2)
        min_dist = dist.min()
        
        agent_distances.append((min_dist, agent_id, agent_data))
    
    # Sort by distance and return top n_agents
    agent_distances.sort(key=lambda x: x[0])
    
    filtered_agents = {}
    for _, agent_id, agent_data in agent_distances[:n_agents]:
        filtered_agents[agent_id] = agent_data
    
    return filtered_agents

def rdp(points, epsilon):
    """
    Recursively simplifies a polyline using the Ramer-Douglas-Peucker algorithm.
    """
    if len(points) < 3:
        return []
    dmax = 0
    index = 0
    end = len(points) - 1
    for i in range(1, end):
        # Perpendicular distance from a point to a line segment
        d = np.linalg.norm(np.cross(np.array(points[end]) - np.array(points[0]), 
                                    np.array(points[0]) - np.array(points[i]))) / np.linalg.norm(np.array(points[end]) - np.array(points[0]))
        if d > dmax:
            index = i
            dmax = d
    
    if dmax > epsilon:
        # Recursive call
        rec_results1 = rdp(points[:index+1], epsilon)
        rec_results2 = rdp(points[index:], epsilon)
        
        # Build the result list
        return rec_results1[:-1] + rec_results2
    else:
        return [points[0], points[end]]

def downsample_map(map_dict, epsilon=0.5):
    """
    Downsamples the road segment points in map_dict using the RDP algorithm.

    Args:
        map_dict (dict): The input map dictionary.
        epsilon (float): The tolerance for the RDP algorithm. A larger value
                         means more aggressive simplification.

    Returns:
        dict: A new dictionary with the same structure but downsampled points.
    """
    processed_map_dict = {}
    for scene_id, segments in map_dict.items():
        processed_segments = []
        for segment in segments:
            # Ensure there are enough points to simplify
            if len(segment['pos_xy']) > 2:
                downsampled_points = rdp(segment['pos_xy'], epsilon)
            else:
                downsampled_points = segment['pos_xy']
            
            processed_segment = {
                'type': segment['type'],
                'pos_xy': downsampled_points
            }
            processed_segments.append(processed_segment)
        processed_map_dict[scene_id] = processed_segments
    return processed_map_dict

def downsample_trajectories(tf_cleaned_traj_dict, frequency_step=5):
    """
    Downsamples agent trajectories by taking every Nth point.

    Args:
        tf_cleaned_traj_dict (dict): The input trajectory dictionary.
        frequency_step (int): The step for downsampling (e.g., 5 for 10Hz to 2Hz).

    Returns:
        dict: A new dictionary with the same structure but downsampled trajectories.
    """
    processed_traj_dict = {}
    for scene_id, agents in tf_cleaned_traj_dict.items():
        processed_agents = {}
        for agent_id, agent_data in agents.items():
            # Downsample using list slicing
            downsampled_traj = agent_data['trajectory'][::frequency_step]
            
            processed_agents[agent_id] = {
                'type': agent_data['type'],
                'trajectory': downsampled_traj
            }
        processed_traj_dict[scene_id] = processed_agents
    return processed_traj_dict

def run_vectorize_process(filename, setting="default"):
    keys_of_interest = [
        'roadgraph_samples/xyz', 
        'roadgraph_samples/dir', 
        'roadgraph_samples/type', 
        'traffic_light_state/current/x',
        'traffic_light_state/current/y',
        'traffic_light_state/current/state',
        'traffic_light_state/past/state',
        'traffic_light_state/future/state',
        'state/past/x',
        'state/past/y',
        'state/past/bbox_yaw',
        'state/current/x',
        'state/current/y',
        'state/current/bbox_yaw',
        'state/future/x',
        'state/future/y',
        'state/future/bbox_yaw',
        'state/type',
        'state/id',
        'scenario/id',
    ]
    data = extract_tfrecord(filename, keys_of_interest, setting=setting)
    tfrecord_name = filename.split('.', 1)[1]
    split = "training"
    if "validation" in filename:
        split = "validation"

    with ProcessPoolExecutor() as executor:
        outputs = list(executor.map(format_dict, data)) # list of tuples (trajectory_dict, obj_orientation)

    list_trajectory_dict = [o[0] for o in outputs]
    list_map_dict = [o[1] for o in outputs]

    tf_map_dict = {}
    for map_data_dict in list_map_dict:
        for sid, map_data in map_data_dict.items():
            tf_map_dict[sid] = map_data

    # Process trajectory dictionaries in parallel
    with ProcessPoolExecutor() as executor:
        tf_cleaned_traj_dicts = []
        for traj_dict in tqdm(list_trajectory_dict, desc='Cleaning Trajectories'):
            tf_cleaned_traj_dicts.append(process_trajectory_dict(traj_dict))

    # Combine the results
    tf_cleaned_traj_dict = {}
    for traj_dict in tf_cleaned_traj_dicts:
        for sid, agent_dict in traj_dict.items():
            if sid not in tf_cleaned_traj_dict:
                tf_cleaned_traj_dict[sid] = {}
            tf_cleaned_traj_dict[sid].update(agent_dict)

    qa_data = []
    qa_file = f"/p/ruishen/processed_waymo_data/language_condition/{split}/waymo_filtered/files/{tfrecord_name}_filter_result.jsonl"
    with open(qa_file) as f:
        for line in f:
            qa_data.append(json.loads(line))

    language_condition_file = f"/p/ruishen/processed_waymo_data/language_condition/{split}/waymo_intent/intent_classification/{tfrecord_name}_intent_language.jsonl"
    language_condition_data = {}
    with open(language_condition_file, 'r') as f:
        for line in f:
            temp_data = json.loads(line)
            sid = temp_data['sid']
            agent_id = temp_data['agent_id']
            language_condition_data[f"{sid}__{agent_id}"] = temp_data['generated_description']

    # filter map and trajectory data by ego traj proximity
    valid_indices = {}
    list_of_valid_map = []
    for qa_dict in tqdm(qa_data, desc='Filtering Valid Indices'):
        sid = qa_dict['sid']
        ego_id = qa_dict['agent_id']
        if sid not in tf_map_dict:
            continue

        try:
            if sid not in valid_indices:
                valid_indices[sid] = {}
                    
            valid_indices[sid][ego_id] = {}
            ego_traj = tf_cleaned_traj_dict[sid][ego_id]['trajectory']

            valid_map_indices = process_map_data(tf_map_dict[sid], ego_traj)
            valid_agent_id = process_traj_data(tf_cleaned_traj_dict[sid], ego_id)
            list_of_valid_map.append({f"{sid}__{ego_id}": {
                'roadgraph_samples/xyz': tf_map_dict[sid]['roadgraph_samples/xyz'][valid_map_indices],
                'roadgraph_samples/dir': tf_map_dict[sid]['roadgraph_samples/dir'][valid_map_indices],
                'roadgraph_samples/type': tf_map_dict[sid]['roadgraph_samples/type'][valid_map_indices],
                # 'traffic_light/xy': list(tf_map_dict[sid]['traffic_light/xy']),
                # 'traffic_light/state': list(tf_map_dict[sid]['traffic_light/state']),
            }})
            valid_indices[sid][ego_id]['valid_agent'] = valid_agent_id
        except Exception as e:
            traceback.print_exc()
            continue

    with ProcessPoolExecutor() as executor:
        road_segments = list(tqdm(
            executor.map(extract_road_segments, list_of_valid_map),
            total=len(list_of_valid_map),
            desc="Extracting road segments"
        ))

    empty_road_segments = []
    for i, road_segment in enumerate(road_segments):
        if len(next(iter(road_segment.values()))) == 0:
            empty_road_segments.append(i)
    for i in sorted(empty_road_segments, reverse=True):
        del road_segments[i]
        del list_of_valid_map[i]

    map_dict = {}
    for map_data in road_segments:
        k = next(iter(map_data.keys()))
        map_dict[k] = map_data[k]


#*******************************************************************************************
#****************************** Start of the main processing *******************************
#*******************************************************************************************


    # Downsample the map and trajectory data
    map_dict = downsample_map(map_dict, epsilon=0.1)
    tf_cleaned_traj_dict = downsample_trajectories(tf_cleaned_traj_dict, frequency_step=5) # downsample from 10Hz to 2Hz

    # Get the testing scenario
    list_of_sid = list(valid_indices.keys())
    testing_sid = list_of_sid[0]
    list_of_ego_agent_id = list(valid_indices[testing_sid].keys())
    testing_ego_id = list_of_ego_agent_id[0]
    other_agent_ids = valid_indices[testing_sid][testing_ego_id]['valid_agent']
    
    # Get testing map and current road
    testing_map = map_dict[f"{testing_sid}__{testing_ego_id}"] # list of road segments, each segment is a dict with keys 'type' and 'pos_xy'

    # Get the testing trajectory and other agents' trajectories
    testing_ego_traj = tf_cleaned_traj_dict[testing_sid][testing_ego_id] # has keys of 'type' and 'trajectory'
    current_road, other_nearby_road = find_nearby_road(testing_ego_traj['trajectory'], testing_map, proximity_threshold=5.0, n_road=6) # has keys 'type' and 'pos_xy'

    other_agent_trajs = {}
    for agent_id in other_agent_ids:
        if agent_id in tf_cleaned_traj_dict[testing_sid]:
            other_agent_trajs[agent_id] = tf_cleaned_traj_dict[testing_sid][agent_id] # each has keys of 'type' and 'trajectory'
    
    other_agent_trajs = find_nearby_agents(other_agent_trajs, testing_ego_traj['trajectory'], n_agents=6) # filter and rank other agents by their distance to the ego trajectory

    return {
        "map_dict": map_dict,
        "tf_cleaned_traj_dict": tf_cleaned_traj_dict,
        "language_condition_data": language_condition_data,
        "valid_indices": valid_indices, # Return this to guide the iteration
        "testing_sid": testing_sid,
        "testing_ego_id": testing_ego_id,
        "testing_map": testing_map,
        "current_road": current_road,
        "testing_ego_traj": testing_ego_traj,       #{"trajectory": (19,2)}
        "other_agent_trajs": other_agent_trajs,
    }


def convert_data(tfrecord_path):
    """
    Converts the Waymo dataset from TFRecord format to a structured format.
    
    Args:
        tfrecord_name (str): The name of the TFRecord file to convert.
        setting (str): The setting for conversion, default is "default".
        
    Returns:
        dict: A dictionary containing the converted data.
    """
    data = run_vectorize_process(filename, "language_condition")

    map_dict = data["map_dict"]
    tf_cleaned_traj_dict = data["tf_cleaned_traj_dict"]
    language_condition_data = data["language_condition_data"]
    valid_indices = data["valid_indices"]

    # for each map in map_dict, find the corresponding ego trajectory and other agents' trajectories
    for sid_egoid, road_segments_list in map_dict.items():
        parts = sid_egoid.split("__")
        sid = parts[0]
        ego_id = int(parts[1])
        ego_traj = tf_cleaned_traj_dict[sid][ego_id]['trajectory']
        print(f"Processing scenario {sid} with ego ID {ego_id} and trajectory {ego_traj}")

        current_road, other_nearby_road = find_nearby_road(ego_traj['trajectory'], road_segments_list, proximity_threshold=5.0, n_road=6) # has keys 'type' and 'pos_xy'

        other_agent_trajs = {}

        other_agent_ids = valid_indices[testing_sid][testing_ego_id]['valid_agent']
        for agent_id in other_agent_ids:
            if agent_id in tf_cleaned_traj_dict[testing_sid]:
                other_agent_trajs[agent_id] = tf_cleaned_traj_dict[testing_sid][agent_id] 
        
        other_agent_trajs = find_nearby_agents(other_agent_trajs, ego_traj['trajectory'], n_agents=6)

        


if __name__ == "__main__":
    filename = "/p/liverobotics/waymo_open_dataset_motion/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord-00000-of-00150"
    data = run_vectorize_process(filename, "language_condition")
    print("Map Dictionary:", data["map_dict"].keys(), "with number:", len(data["map_dict"].keys()))
    print("Trajectory Dictionary:", data["tf_cleaned_traj_dict"].keys())
    print("the first trajector in tf_cleaned_traj_dict:", list(data["tf_cleaned_traj_dict"].values())[0].keys())
    print("Language Condition Data:", data["language_condition_data"])
    breakpoint()
    print("Testing Scenario ID:", data["testing_sid"])
    print("Testing Ego Agent ID:", data["testing_ego_id"])
    print("Testing Map:", data["testing_map"])
    print("Current Road:", data["current_road"])
    print("Testing Ego Trajectory:", data["testing_ego_traj"])
    print("Other Agent Trajectories:", data["other_agent_trajs"])
    breakpoint()  # To inspect the data interactively
    max_len = 0
    
    # Iterate through each list of road segments in the map_dict values
    for key,road_segments_list in data["map_dict"].items():
        # Iterate through each individual segment dictionary in the list
        for segment in road_segments_list:
            # Check if the 'pos_xy' key exists and its value is a list
            if 'pos_xy' in segment and isinstance(segment['pos_xy'], list):
                # Get the number of arrays (points) in the current pos_xy list
                current_len = len(segment['pos_xy'])
                # If this length is greater than the max found so far, update max_len
                if current_len > max_len:
                    max_len = current_len
                    print("the longest array is:", segment['pos_xy'],"with road type:", segment['type'])
                    print(key)
                    parts = key.split("__")
                    testing_sid = parts[0]
                    testing_ego_id = int(parts[1])
                    ego_traj = data["tf_cleaned_traj_dict"][testing_sid][testing_ego_id]['trajectory']
                    print("ego trajectory:", ego_traj)


    print(f"After iterating through all maps, the longest 'pos_xy' list contains {max_len} arrays.")
    
    # --- End of the requested loop ---

    print("\nMap Dictionary Keys:", data["map_dict"].keys(), "with number:", len(data["map_dict"].keys()))
    
    



















