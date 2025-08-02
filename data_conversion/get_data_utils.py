from scipy.spatial import cKDTree
import numpy as np
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
import tensorflow as tf

ROAD_TYPE = {
    1: "LaneCenter-Freeway",
    2: "LaneCenter-SurfaceStreet", 
    3: "LaneCenter-BikeLane",
    6: "RoadLine-BrokenSingleWhite",
    7: "RoadLine-SolidSingleWhite",
    8: "RoadLine-SolidDoubleWhite", 
    9: "RoadLine-BrokenSingleYellow",
    10: "RoadLine-BrokenDoubleYellow",
    11: "Roadline-SolidSingleYellow",
    12: "Roadline-SolidDoubleYellow",
    13: "RoadLine-PassingDoubleYellow",
    15: "RoadEdgeBoundary",
    16: "RoadEdgeMedian",
    17: "StopSign",
    18: "Crosswalk",
    19: "SpeedBump"
}

def trim_invalid_ends(points, directions):
    """
    Remove invalid states from the beginning and end of the sequence.
    """
    valid_mask = ~np.all(points == -1, axis=1)
    if not np.any(valid_mask):
        return np.array([]), np.array([])
        
    first_valid = np.where(valid_mask)[0][0]
    last_valid = np.where(valid_mask)[0][-1]
    
    return points[first_valid:last_valid + 1], directions[first_valid:last_valid + 1]

def interpolate_invalid_states(points, directions, max_gap=3):
    """Interpolate small gaps of invalid states in the middle of valid states."""
    valid_mask = ~np.all(points == -1, axis=1)
    
    # Find runs of invalid states
    invalid_runs = np.where(~valid_mask)[0]
    if len(invalid_runs) == 0:
        return points, directions
    
    # Split into consecutive runs
    run_starts = [invalid_runs[0]]
    run_lengths = []
    current_length = 1
    
    for i in range(1, len(invalid_runs)):
        if invalid_runs[i] == invalid_runs[i-1] + 1:
            current_length += 1
        else:
            run_lengths.append(current_length)
            run_starts.append(invalid_runs[i])
            current_length = 1
    run_lengths.append(current_length)
    
    # Interpolate gaps that are small enough and have valid points on both sides
    points = points.copy()
    directions = directions.copy()
    
    for start, length in zip(run_starts, run_lengths):
        if length > max_gap:
            continue
            
        # Find valid points before and after
        before_idx = start - 1
        after_idx = start + length
        
        if before_idx < 0 or after_idx >= len(points):
            continue
        
        if np.all(points[before_idx] == -1) or np.all(points[after_idx] == -1):
            continue
            
        # Interpolate positions
        for i in range(length):
            alpha = (i + 1) / (length + 1)
            points[start + i] = (1 - alpha) * points[before_idx] + alpha * points[after_idx]
            directions[start + i] = (1 - alpha) * directions[before_idx] + alpha * directions[after_idx]
            # Normalize interpolated direction
            if np.any(directions[start + i] != 0):
                directions[start + i] = directions[start + i] / np.linalg.norm(directions[start + i])
    
    return points, directions

def extract_road_segments(map_dict, threshold=2.0):
    """
    Extract road segments from map points and categories.
    Returns a list of dictionaries containing road type, positions, and directions.
    """
    sid = list(map_dict.keys())[0]
    map_points = map_dict[sid]["roadgraph_samples/xyz"]
    road_categories = map_dict[sid]["roadgraph_samples/type"]
    road_dir = map_dict[sid]["roadgraph_samples/dir"]

    segments = []
    
    # Extract x and y coordinates, ignoring z
    xy_coords = map_points[:, :2]
    
    for category in ROAD_TYPE.keys():
        # Select points of the current category
        indices = np.where(road_categories == category)[0]
        
        if len(indices) > 0:
            category_points = xy_coords[indices]
            category_dirs = road_dir[indices]
            
            # Clean up invalid states
            clean_points, clean_dirs = trim_invalid_ends(category_points, category_dirs)
            if len(clean_points) == 0:
                continue
                
            clean_points, clean_dirs = interpolate_invalid_states(clean_points, clean_dirs)
            
            current_segment_points = []
            current_segment_dirs = []

            # Loop through points to form continuous segments
            for i in range(len(clean_points) - 1):
                current_segment_points.append(clean_points[i])
                current_segment_dirs.append(clean_dirs[i])
                
                # Calculate the distance to the next point
                distance = np.linalg.norm(clean_points[i + 1] - clean_points[i])
                
                # If the points are too far apart, end current segment
                if distance > threshold:
                    if len(current_segment_points) > 3:
                        segments.append({
                            "type": ROAD_TYPE[category],
                            "pos_xy": np.array(current_segment_points),
                            "dir": np.array(current_segment_dirs)
                        })
                    current_segment_points = []
                    current_segment_dirs = []

            # Add the last point and segment
            if len(clean_points) > 0:
                current_segment_points.append(clean_points[-1])
                current_segment_dirs.append(clean_dirs[-1])
                if len(current_segment_points) > 3:
                    segments.append({
                        "type": ROAD_TYPE[category],
                        "pos_xy": np.array(current_segment_points),
                        "dir": np.array(current_segment_dirs)
                    })

    return {sid: segments}

def format_dict(data):
    if 'roadgraph_samples/xyz' in data:
        data['roadgraph_samples/xyz'] = data['roadgraph_samples/xyz'].reshape((30000, 3))
    
    if 'roadgraph_samples/dir' in data:
        data['roadgraph_samples/dir'] = data['roadgraph_samples/dir'].reshape((30000, 3))
    
    if 'traffic_light_state/past/state' in data:
        data['traffic_light_state/past/state'] = data['traffic_light_state/past/state'].reshape((10, 16))
        data['traffic_light_state/future/state'] = data['traffic_light_state/future/state'].reshape((80, 16))
        light_states = np.concatenate((data['traffic_light_state/past/state'], np.array([data['traffic_light_state/current/state']]), data['traffic_light_state/future/state']), axis=0)  # Shape: (91, 16)

    if 'traffic_light_state/past/x' in data:
        data['traffic_light_state/past/x'] = data['traffic_light_state/past/x'].reshape((10, 16))
        data['traffic_light_state/past/y'] = data['traffic_light_state/past/y'].reshape((10, 16))
        data['traffic_light_state/future/x'] = data['traffic_light_state/future/x'].reshape((80, 16))
        data['traffic_light_state/future/y'] = data['traffic_light_state/future/y'].reshape((80, 16))
        light_xy_pos = np.stack((data['traffic_light_state/current/x'], data['traffic_light_state/current/y']), axis=-1)  # Shape: (16, 2)

    
    if 'state/past/x' in data:
        data['state/past/x'] = data['state/past/x'].reshape((128, 10))
        data['state/past/y'] = data['state/past/y'].reshape((128, 10))
        data['state/future/x'] = data['state/future/x'].reshape((128, 80))
        data['state/future/y'] = data['state/future/y'].reshape((128, 80))
        
        # Combining arrays to form the trajectory (91 time steps)
        past_xy = np.stack((data['state/past/x'], data['state/past/y']), axis=-1)  # Shape: (128, 10, 2)
        current_xy = np.stack((data['state/current/x'], data['state/current/y']), axis=-1)[:, np.newaxis, :]  # Shape: (128, 1, 2)
        future_xy = np.stack((data['state/future/x'], data['state/future/y']), axis=-1)  # Shape: (128, 80, 2)

        full_trajectory = np.concatenate((past_xy, current_xy, future_xy), axis=1)  # Shape: (128, 91, 2)
    
    if 'state/past/bbox_yaw' in data:
        data['state/past/bbox_yaw'] = data['state/past/bbox_yaw'].reshape((128, 10))
        data['state/current/bbox_yaw'] = data['state/current/bbox_yaw'].reshape((128, 1))
        data['state/future/bbox_yaw'] = data['state/future/bbox_yaw'].reshape((128, 80))

        obj_orientation = np.concatenate((data['state/past/bbox_yaw'], data['state/current/bbox_yaw'], data['state/future/bbox_yaw']), axis=1)  # Shape: (128, 91)

    # Dictionary to store the type + index as keys and the trajectory as values
    type_dict = {
        0: 'Unset',
        1: 'Vehicle',
        2: 'Pedestrian',
        3: 'Cyclist',
        4: 'Other'
    }
    trajectory_dict = {}
    obj_orientation_dict = {}

    # Populate the dictionary
    for i in range(128):
        if data['state/id'][i] < 0:
            continue
        if data['state/type'][i] < 0:
            continue
        agent_id = int(data['state/id'][i])
        obj_type = type_dict[int(data['state/type'][i])]
        
        # Add to dictionary
        trajectory_dict[agent_id] = {"trajectory": full_trajectory[i],
                                     "heading": obj_orientation[i],
                                "type": obj_type}
        obj_orientation_dict[agent_id] = obj_orientation[i]
    
    result_trajectory_dict = {data["scenario/id"]: trajectory_dict}

    map_dict = {
        data["scenario/id"]:
        {
            "obj_orientation": obj_orientation_dict,
            "roadgraph_samples/xyz": data["roadgraph_samples/xyz"],
            "roadgraph_samples/dir": data["roadgraph_samples/dir"],
            "roadgraph_samples/type": data["roadgraph_samples/type"],
            # "traffic_light/xy": light_xy_pos,
            # "traffic_light/state": light_states
        }
    }

    return result_trajectory_dict, map_dict 

def extract_tfrecord(tfrecord_file, interested_keys, setting="default"):
    """Extract data from TFRecord file"""
    # Create a dataset from the TFRecord file
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
    records_list = []
    # Take the first record from the dataset
    for raw_record in raw_dataset:
        # Parse the example using tf.train.Example
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())

        # Create a dictionary to store the data
        data_dict = {}

        # Iterate through all features in the example
        for key, feature in example.features.feature.items():
            if key in interested_keys:
                # Determine the type of the feature and extract the value
                if feature.HasField('bytes_list'):
                    byte_values = feature.bytes_list.value
                    if len(byte_values) == 1:
                        # Single string case
                        value = byte_values[0].decode('utf-8')
                    else:
                        # List of strings case
                        value = [item.decode('utf-8') for item in byte_values]
                            
                elif feature.HasField('float_list'):
                    # Float feature
                    value = np.array(feature.float_list.value)
                elif feature.HasField('int64_list'):
                    # Integer feature
                    value = np.array(feature.int64_list.value)
                else:
                    # Unknown type
                    value = None

                # Add to the dictionary
                data_dict[key] = value
        
        records_list.append(data_dict)

    if setting == "default":
        records_list = records_list[:50]
    elif setting == "language_condition":
        records_list = records_list[50:100]

    return records_list

def create_ellipse_polygon(p1, p2, radius, ellipse_ratio=0.5, num_points=16):
    """Create ellipse polygon with width=2*radius aligned to direction vector p1->p2."""
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length == 0:
        return
        
    # Normalize direction vector
    direction = direction / length
    p1 = p1 + 0.5 * direction * radius
    # Get perpendicular vector
    perp = np.array([-direction[1], direction[0]])

    angles = np.linspace(0, 2*np.pi, num_points)
    ellipse_points = []
    for angle in angles:
        # Create elongated horizontal ellipse
        x = radius * np.cos(angle)  # Major axis
        y = radius * ellipse_ratio * np.sin(angle)  # Minor axis
        # Rotate to align with direction vector and translate to center
        point = p1 + x*direction + y*perp
        ellipse_points.append((point[0], point[1]))
    
    return Polygon(ellipse_points)

def trajectory_proximity_filter(input_array, trajectory, mode='filter', 
                                shape='circle', radius=10.0, ellipse_ratio=0.5):
    '''Efficient proximity filtering using spatial data structures'''
    input_array = np.asarray(input_array)
    trajectory = np.asarray(trajectory)
    
    if shape == 'circle':
        # Use KD-tree for efficient circular proximity
        tree = cKDTree(trajectory)
        distances, _ = tree.query(input_array, k=1)
        valid_points = distances <= radius
        
    else:  # ellipse mode
        # Create union of ellipse polygons along trajectory
        ellipses = []
        for i in range(len(trajectory)-1):
            ellipse = create_ellipse_polygon(
                trajectory[i], trajectory[i+1], 
                radius, ellipse_ratio
            )
            if ellipse is not None:
                ellipses.append(ellipse)

        valid_area = unary_union(ellipses)
        valid_points = np.array([
            valid_area.contains(Point(p)) 
            for p in input_array
        ])
    
    if mode == 'check':
        return np.any(valid_points)
    else:
        return valid_points

def process_map_data(map_data, ego_traj):
    map_points = map_data['roadgraph_samples/xyz']
    valid_points = trajectory_proximity_filter(map_points, ego_traj, mode='filter', shape='ellipse')
    return valid_points