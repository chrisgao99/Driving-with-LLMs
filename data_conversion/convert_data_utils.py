import numpy as np
from data_conversion.vector_utils_waymo_custom import VehicleField, PedestrianField, RoadField, EgoField


# def get_heading(trajectory):
#     """
#     Computes the heading of a trajectory based on its first two points.

#     Args:
#         trajectory (list): A list of [x, y] points representing the trajectory.

#     Returns:
#         np.array: An array of headings for each segment of the trajectory in radians.
#     """
#     if len(trajectory) < 2:
#         return 0.0
#     trajectory = np.array(trajectory)
#     dx = trajectory[1:, 0] - trajectory[:-1, 0]
#     dy = trajectory[1:, 1] - trajectory[:-1, 1]
#     headings = np.arctan2(dy, dx)
#     return headings,dx, dy

# def get_speed(trajectory, dt=0.5):
#     """
#     Computes the speed of a trajectory based on its points.

#     Args:
#         trajectory (list): A list of [x, y] points representing the trajectory.
#         dt (float): The time step between points in seconds.

#     Returns:
#         np.array: An array of speeds for each segment of the trajectory.
#     """
#     if len(trajectory) < 2:
#         return 0.0
#     trajectory = np.array(trajectory)
#     dx = np.diff(trajectory[:, 0])
#     dy = np.diff(trajectory[:, 1])
#     distances = np.sqrt(dx**2 + dy**2)
#     speeds = distances / dt
#     return speeds

# def get_acceleration(trajectory, dt=0.5):
#     """
#     Computes the acceleration of a trajectory based on its points.

#     Args:
#         trajectory (list): A list of [x, y] points representing the trajectory.
#         dt (float): The time step between points in seconds.

#     Returns:
#         np.array: An array of accelerations for each segment of the trajectory.
#     """
#     if len(trajectory) < 3:
#         return 0.0
#     speeds = get_speed(trajectory, dt)
#     accelerations = np.diff(speeds) / dt
#     return accelerations


def _interpolate_invalid_points(trajectory):
    """
    Fills invalid [-1, -1] points in a trajectory using linear interpolation
    or extrapolation.
    """
    # Make a copy to avoid modifying the original array
    traj_filled = np.copy(trajectory.astype(float))
    n_points = len(traj_filled)
    is_invalid = np.all(traj_filled == -1, axis=1)
    valid_indices = np.where(~is_invalid)[0]

    # Cannot process if there are fewer than 2 valid points to define a line
    if len(valid_indices) < 2:
        return traj_filled

    # Find and process each continuous block of invalid points
    i = 0
    while i < n_points:
        if is_invalid[i]:
            start_block = i
            end_block = i
            while end_block + 1 < n_points and is_invalid[end_block + 1]:
                end_block += 1
            
            # Find the last valid point before the block
            prev_valid_idx = valid_indices[valid_indices < start_block][-1] if any(valid_indices < start_block) else -1
            # Find the first valid point after the block
            next_valid_idx = valid_indices[valid_indices > end_block][0] if any(valid_indices > end_block) else -1

            # Case 1: Interpolate (gap is between two valid points)
            if prev_valid_idx != -1 and next_valid_idx != -1:
                p0 = traj_filled[prev_valid_idx]
                p1 = traj_filled[next_valid_idx]
                time_gap = next_valid_idx - prev_valid_idx
                for j in range(start_block, end_block + 1):
                    # Calculate how far into the gap this point is
                    alpha = (j - prev_valid_idx) / time_gap
                    traj_filled[j] = p0 + alpha * (p1 - p0)

            # Case 2: Extrapolate (gap is at the beginning)
            elif next_valid_idx != -1:
                p1 = traj_filled[next_valid_idx]
                p2 = traj_filled[valid_indices[valid_indices > next_valid_idx][0]]
                # Velocity is change in position per time step
                velocity = p2 - p1
                for j in range(start_block, end_block + 1):
                    time_diff = next_valid_idx - j
                    traj_filled[j] = p1 - (velocity * time_diff)

            # Case 3: Extrapolate (gap is at the end)
            elif prev_valid_idx != -1:
                p0 = traj_filled[prev_valid_idx]
                p_minus_1 = traj_filled[valid_indices[valid_indices < prev_valid_idx][-1]]
                velocity = p0 - p_minus_1
                for j in range(start_block, end_block + 1):
                    time_diff = j - prev_valid_idx
                    traj_filled[j] = p0 + (velocity * time_diff)
            
            i = end_block # Move index past the block we just processed
        i += 1
            
    return traj_filled



def _process_agent_trajectory_at_time(trajectory, t, dt=0.5):
    """
    Calculates dynamic properties for a single agent at a specific time step t.
    """
    # If current position is invalid, agent is not in the scene. Return all zeros.
    if np.all(trajectory[t] == -1):
        return 0, 0, 0, 0, 0, 0, 0  # accel, speed, x, y, heading, dx, dy

    x, y = trajectory[t]
    speed, heading, dx, dy, accel = 0.0, 0.0, 0.0, 0.0, 0.0

    # Speed and heading require a valid previous point (t-1)
    if t > 0 and not np.all(trajectory[t-1] == -1):
        p_curr = trajectory[t]
        p_prev = trajectory[t-1]
        
        dx = p_curr[0] - p_prev[0]
        dy = p_curr[1] - p_prev[1]
        
        distance = np.sqrt(dx**2 + dy**2)
        speed = distance / dt
        heading = np.arctan2(dy, dx)

        # Acceleration requires a valid point at t-2 to calculate previous speed
        if t > 1 and not np.all(trajectory[t-2] == -1):
            p_prev2 = trajectory[t-2]
            dx_prev = p_prev[0] - p_prev2[0]
            dy_prev = p_prev[1] - p_prev2[1]
            speed_prev = np.sqrt(dx_prev**2 + dy_prev**2) / dt
            accel = (speed - speed_prev) / dt
            
    return accel, speed, x, y, heading, dx, dy


def convert_to_descriptor_format(data):
    """
    Given
    Converts raw scenario data into a list of 19 dictionaries, one for each
    time step, in chronological order.
    """
    # --- PRE-PROCESS TRAJECTORIES ---
    # Call the new helper function to fill in [-1, -1] values first.
    ego_traj = _interpolate_invalid_points(data['Ego Trajectory']['trajectory'])
    
    nearby_agents = data.get('Nearby Agent Trajectories', {})
    processed_nearby_agents = {}
    for agent_id, agent_data in nearby_agents.items():
        processed_data = agent_data.copy()
        processed_data['trajectory'] = _interpolate_invalid_points(agent_data['trajectory'])
        processed_nearby_agents[agent_id] = processed_data

    list_of_converted_data = []
    
    map_data = data.get('Map Data', [])
    num_road_segments = min(len(map_data), 60)
    
    road_type_map = {
        'LaneCenter-Freeway': RoadField.TYPE_LANE_CENTER_FREEWAY, 'LaneCenter-SurfaceStreet': RoadField.TYPE_LANE_CENTER_SURFACE_STREET,
        'LaneCenter-BikeLane': RoadField.TYPE_LANE_CENTER_BIKE_LANE, 'RoadLine-BrokenSingleWhite': RoadField.TYPE_ROAD_LINE_BROKEN_SINGLE_WHITE,
        'RoadLine-SolidSingleWhite': RoadField.TYPE_ROAD_LINE_SOLID_SINGLE_WHITE, 'RoadLine-SolidDoubleWhite': RoadField.TYPE_ROAD_LINE_SOLID_DOUBLE_WHITE,
        'RoadLine-BrokenSingleYellow': RoadField.TYPE_ROAD_LINE_BROKEN_SINGLE_YELLOW, 'RoadLine-BrokenDoubleYellow': RoadField.TYPE_ROAD_LINE_BROKEN_DOUBLE_YELLOW,
        'RoadLine-SolidSingleYellow': RoadField.TYPE_ROAD_LINE_SOLID_SINGLE_YELLOW, 'RoadLine-SolidDoubleYellow': RoadField.TYPE_ROAD_LINE_SOLID_DOUBLE_YELLOW,
        'RoadLine-PassingDoubleYellow': RoadField.TYPE_ROAD_LINE_PASSING_DOUBLE_YELLOW, 'RoadEdgeBoundary': RoadField.TYPE_ROAD_EDGE_BOUNDARY,
        'RoadEdge-Median': RoadField.TYPE_ROAD_EDGE_MEDIAN, 'StopSign': RoadField.TYPE_STOP_SIGN, 'Crosswalk': RoadField.TYPE_CROSSWALK, 'SpeedBump': RoadField.TYPE_SPEED_BUMP,
    }

    # Loop from t=0 to t=18 to generate snapshots in chronological order
    for t in range(19):
        ego_descriptor = np.zeros(len(EgoField), dtype=np.float32)
        vehicle_descriptors = np.zeros((30, len(VehicleField)), dtype=np.float32)
        pedestrian_descriptors = np.zeros((20, len(PedestrianField)), dtype=np.float32)
        road_descriptors = np.zeros((60, len(RoadField)), dtype=np.float32)

        # --- 1. Process Ego Vehicle at time t ---
        accel, speed, x, y, heading, dx, dy = _process_agent_trajectory_at_time(ego_traj, t)
        ego_descriptor[EgoField.ACCEL], ego_descriptor[EgoField.SPEED] = accel, speed
        ego_descriptor[EgoField.X], ego_descriptor[EgoField.Y] = x, y
        ego_descriptor[EgoField.HEADING], ego_descriptor[EgoField.DX], ego_descriptor[EgoField.DY] = heading, dx, dy
        
        current_ego_pos = np.array([x, y])

        # --- 2. Process Nearby Agents at time t ---
        vehicle_count = 0
        for agent_data in nearby_agents.values():
            if vehicle_count >= 30: break
            accel, speed, x, y, heading, dx, dy = _process_agent_trajectory_at_time(agent_data['trajectory'], t)
            
            desc = vehicle_descriptors[vehicle_count]
            desc[VehicleField.SPEED], desc[VehicleField.X], desc[VehicleField.Y] = speed, x, y
            desc[VehicleField.DX], desc[VehicleField.DY], desc[VehicleField.HEADING] = dx, dy, heading
            vehicle_count += 1
            
        # --- 3. Process Road Descriptors relative to ego at time t ---
        for i in range(num_road_segments):
            segment = map_data[i]
            points = np.array(segment['pos_xy'])
            if len(points) > 10:
                distances = np.linalg.norm(points - current_ego_pos, axis=1)
                closest_idx = np.argmin(distances)
                start_idx = min(closest_idx + 10, len(points)) - 10
                selected_points = points[start_idx : start_idx + 10]
            else:
                selected_points = points
            
            for j in range(10):
                if j < len(selected_points):
                    road_descriptors[i, 2*j:2*j+2] = selected_points[j]
                else:
                    road_descriptors[i, RoadField.INVALID_1 + j] = 1.0
            
            segment_type_str = segment.get('type')
            if segment_type_str in road_type_map:
                road_descriptors[i, road_type_map[segment_type_str]] = 1.0

        # --- 4. Assemble dictionary for the current time step and append to list ---
        data_at_t = {
            "ego_vehicle_descriptor": ego_descriptor,
            "vehicle_descriptors": vehicle_descriptors,
            "pedestrian_descriptors": pedestrian_descriptors,
            "road_descriptors": road_descriptors
        }
        list_of_converted_data.append(data_at_t)

    return list_of_converted_data

def get_qa_for_descriptor(list_of_converted_data):


    return 0


if __name__ == '__main__':
    # This is the sample data of one map to test the conversion function
    # one sample data will be converted to a list of 19 dicts, one for each time step
    sample_data = {
        'Map Data': [{'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1963.56982422, 7806.85546875]), np.array([1963.52709961, 7843.18310547])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1966.96679688, 7806.84423828]), np.array([1966.98547363, 7843.69970703])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1966.93896484, 7785.59375]), np.array([1966.95495605, 7793.99853516]), np.array([1969.23913574, 7797.90966797]), np.array([1970.70812988, 7798.00439453])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1966.94824219, 7793.01367188]), np.array([1966.96679688, 7806.84423828])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1963.64428711, 7792.94677734]), np.array([1963.56982422, 7806.85546875])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1970.98742676, 7801.27636719]), np.array([1969.01513672, 7801.32470703]), np.array([1967.06616211, 7801.60839844]), np.array([1963.73632812, 7804.89355469]), np.array([1963.56982422, 7806.85546875])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1970.98742676, 7801.27636719]), np.array([1962.15856934, 7801.75683594])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1967.05712891, 7804.48193359]), np.array([1966.96679688, 7806.84423828])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1970.98742676, 7801.27636719]), np.array([1967.02539062, 7801.43017578]), np.array([1962.11303711, 7802.08984375])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1970.98742676, 7801.27636719]), np.array([1968.0501709, 7801.18896484]), np.array([1962.22180176, 7800.45507812])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1963.92077637, 7759.53564453]), np.array([1966.51672363, 7776.18798828]), np.array([1966.83496094, 7780.14111328]), np.array([1966.93896484, 7785.59375])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1963.57446289, 7760.57958984]), np.array([1963.64428711, 7792.94677734])]}, {'type': 'RoadLine-BrokenSingleWhite', 'pos_xy': [np.array([1961.87744141, 7761.74658203]), np.array([1961.93127441, 7802.21337891])]}, {'type': 'RoadLine-SolidSingleWhite', 'pos_xy': [np.array([1965.40332031, 7785.50488281]), np.array([1965.16125488, 7843.51123047])]}, {'type': 'RoadEdgeBoundary', 'pos_xy': [np.array([1968.43725586, 7822.7265625]), np.array([1968.78320312, 7845.12158203])]}, {'type': 'RoadEdgeBoundary', 'pos_xy': [np.array([1968.4642334, 7806.51367188]), np.array([1968.43725586, 7822.7265625])]}, {'type': 'RoadEdgeBoundary', 'pos_xy': [np.array([1967.46484375, 7760.70117188]), np.array([1968.63195801, 7782.35400391]), np.array([1968.51171875, 7794.00048828]), np.array([1971.43725586, 7794.03027344])]}, {'type': 'Crosswalk', 'pos_xy': [np.array([1965.60144043, 7845.03515625]), np.array([1968.67199707, 7845.16992188])]}],
        'Ego Trajectory': {'type': 'Vehicle', 'trajectory': np.array([[-1.00000000e+00, -1.00000000e+00], [1.96526404e+03, 7.76571191e+03], [1.96557971e+03, 7.77292139e+03], [1.96608276e+03, 7.78193604e+03], [1.96643823e+03, 7.78937842e+03], [1.96669189e+03, 7.79597900e+03], [1.96683398e+03, 7.80195312e+03], [1.96690027e+03, 7.80714648e+03], [1.96693176e+03, 7.81174316e+03], [1.96695154e+03, 7.81586426e+03], [1.96697522e+03, 7.81946924e+03], [1.96698767e+03, 7.82249658e+03], [1.96699011e+03, 7.82487500e+03], [1.96699292e+03, 7.82656885e+03], [1.96699207e+03, 7.82779785e+03], [1.96698657e+03, 7.82872119e+03], [1.96698218e+03, 7.82952588e+03], [1.96697607e+03, 7.83041113e+03], [1.96697656e+03, 7.83131445e+03]])},
        'Nearby Agent Trajectories': {650: {'type': 'Vehicle', 'trajectory': np.array([[1966.87072754, 7828.43798828], [1966.87475586, 7828.45458984], [1966.86901855, 7828.484375], [1966.86706543, 7828.66162109], [1966.86401367, 7829.0703125], [1966.85656738, 7829.80566406], [1966.84875488, 7830.76269531], [1966.83544922, 7831.95849609], [1966.8236084, 7833.11181641], [1966.81616211, 7834.24609375], [1966.81835938, 7835.26904297], [1966.81335449, 7836.15917969], [1966.80432129, 7836.77832031], [1966.7989502, 7837.21044922], [1966.81237793, 7837.69873047], [1966.82092285, 7838.26416016], [1966.83056641, 7838.88818359], [1966.83239746, 7839.64648438], [1966.84216309, 7840.64794922]])}, 672: {'type': 'Vehicle', 'trajectory': np.array([[1964.07141113, 7790.80615234], [1964.06347656, 7795.94921875], [1964.04870605, 7800.78564453], [1964.0435791, 7805.06054688], [1964.01977539, 7809.04199219], [1964.00061035, 7812.54638672], [1963.9921875, 7815.69042969], [1963.96240234, 7818.52832031], [1963.95715332, 7821.07714844], [1963.95263672, 7823.296875], [1963.94311523, 7825.2109375], [1963.94128418, 7826.83349609], [1963.94030762, 7828.14355469], [1963.94372559, 7829.20605469], [1963.94592285, 7830.00097656], [1963.94299316, 7830.61865234], [1963.94763184, 7831.14111328], [1963.94812012, 7831.54931641], [1963.9543457, 7831.82324219]])}}
    }
    
    # Run the conversion
    converted_data = convert_to_descriptor_format(sample_data)

    converted_data = converted_data[1]  # Get the first time step for demonstration

    # Print shapes and a sample of the data to verify
    print("--- CONVERSION COMPLETE ---")
    for key, value in converted_data.items():
        print(f"Key: '{key}'")
        print(f"Shape: {value.shape}")
    
    print("\n--- Sample: Ego Descriptor ---")
    print(converted_data['ego_vehicle_descriptor'])

    print("\n--- Sample: First Vehicle Descriptor ---")
    print(converted_data['vehicle_descriptors'])

    print("\n--- Sample: First 3 Road in Road Descriptor ---")
    print(converted_data['road_descriptors'][0, :20])
    print(converted_data['road_descriptors'][0, 20:30])
    print(converted_data['road_descriptors'][0, 30:])
    print(converted_data['road_descriptors'][1, :20])
    print(converted_data['road_descriptors'][1, 20:30])
    print(converted_data['road_descriptors'][1, 30:])
    print(converted_data['road_descriptors'][2, :20])
    print(converted_data['road_descriptors'][2, 20:30])
    print(converted_data['road_descriptors'][2, 30:])

    print("\n--- Sample: First Pedestrian Descriptor ---")
    print(converted_data['pedestrian_descriptors'])