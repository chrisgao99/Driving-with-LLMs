import math
import numpy as np
import textwrap
from get_data_for_eval import run_vectorize_process, find_nearby_road, find_nearby_agents

class WaypointAgent:
    """
    An agent that processes raw vehicle trajectory data, generates a detailed
    natural language prompt describing the driving scene, and asks an LLM
    to predict the next waypoint for the ego vehicle.
    """

    def __init__(self):
        """Initializes the WaypointAgent."""
        self.delimiter = "####"

    def _calculate_dynamics(self, trajectory: np.ndarray, current_time_idx: int, dt: float = 0.5):
        """
        Calculates the dynamic state (speed, heading, acceleration) for a single
        agent at a specific time from its trajectory data.

        Args:
            trajectory (np.ndarray): The agent's trajectory, an array of [x, y] points.
            current_time_idx (int): The index for the current time step.
            dt (float): The time delta between trajectory points in seconds.

        Returns:
            tuple: A tuple containing (speed, heading_degrees, acceleration).
                   Returns (0, 0, 0) if data is insufficient.
        """
        # Ensure we have at least two points to calculate speed and heading
        if current_time_idx < 1 or np.all(trajectory[current_time_idx] == -1) or np.all(trajectory[current_time_idx - 1] == -1):
            return 0.0, 0.0, 0.0

        p_curr = trajectory[current_time_idx]
        p_prev = trajectory[current_time_idx - 1]

        dx = p_curr[0] - p_prev[0]
        dy = p_curr[1] - p_prev[1]

        distance = math.sqrt(dx**2 + dy**2)
        speed = distance / dt
        heading_rad = math.atan2(dy, dx)
        heading_deg = math.degrees(heading_rad)

        acceleration = 0.0
        # Ensure we have three points to calculate acceleration
        if current_time_idx >= 2 and not np.all(trajectory[current_time_idx - 2] == -1):
            p_prev2 = trajectory[current_time_idx - 2]
            dx_prev = p_prev[0] - p_prev2[0]
            dy_prev = p_prev[1] - p_prev2[1]
            speed_prev = math.sqrt(dx_prev**2 + dy_prev**2) / dt
            acceleration = (speed - speed_prev) / dt
        
        return speed, heading_deg, acceleration

    def generate_waypoint_prompt(self, scene_data: dict, current_time_idx: int) -> str:
        """
        Constructs a detailed prompt for an LLM to predict the next waypoint.

        Args:
            scene_data (dict): The raw dictionary containing map and trajectory data.
            current_time_idx (int): The current time step index (e.g., 10 for the 11th step).
            driving_intention (str): A high-level goal for the ego vehicle.

        Returns:
            str: The fully formatted prompt ready to be sent to an LLM.
        """
        
        # --- 1. Process Ego Vehicle ---
        ego_traj = scene_data['Ego Trajectory']['trajectory']
        ego_x, ego_y = ego_traj[current_time_idx]
        ego_speed, ego_heading, ego_accel = self._calculate_dynamics(ego_traj, current_time_idx)

        ego_state_desc = textwrap.dedent(f"""\
            - Your current position (x, y) is ({ego_x:.2f}, {ego_y:.2f}).
            - Your speed is {ego_speed:.2f} m/s.
            - Your heading is {ego_heading:.2f} degrees.
            - Your acceleration is {ego_accel:.2f} m/s^2.""")

        # --- 2. Process Surrounding Vehicles ---
        surrounding_vehicles_desc = []
        for agent_id, agent_data in scene_data.get('Nearby Agent Trajectories', {}).items():
            agent_traj = agent_data['trajectory']
            
            # Check if agent is valid at the current time
            if current_time_idx >= len(agent_traj) or np.all(agent_traj[current_time_idx] == -1):
                continue

            agent_x, agent_y = agent_traj[current_time_idx]
            distance_to_ego = math.sqrt((ego_x - agent_x)**2 + (ego_y - agent_y)**2)

            # Only include vehicles within a 100m radius
            if distance_to_ego > 100:
                continue

            speed, heading, _ = self._calculate_dynamics(agent_traj, current_time_idx)
            
            surrounding_vehicles_desc.append(
                f"- Vehicle `{agent_id}` is {distance_to_ego:.2f} meters away at position ({agent_x:.2f}, {agent_y:.2f}).\n"
                f"  It is traveling at {speed:.2f} m/s with a heading of {heading:.2f} degrees."
            )
        
        if not surrounding_vehicles_desc:
            surrounding_vehicles_desc.append("No other vehicles are nearby.")

        # --- 3. Process Road Layout ---
        road_layout_desc = []
        for road_segment in scene_data.get('Map Data', []):
            points = np.array(road_segment['pos_xy'])
            if points.size == 0:
                continue

            distances_to_ego = np.linalg.norm(points - np.array([ego_x, ego_y]), axis=1)
            min_dist = np.min(distances_to_ego)

            # Only describe road features within 20 meters
            if min_dist < 20:
                road_type = road_segment['type'].replace('-', ' ').replace('RoadLine', 'Road Line')
                points_str = ", ".join([f"({p[0]:.2f}, {p[1]:.2f})" for p in points])
                road_layout_desc.append(
                    f"- A '{road_type}' feature is nearby with the following points:\n"
                    f"  Coordinates: [{points_str}]"
                )

        if not road_layout_desc:
            road_layout_desc.append("No relevant map features are nearby.")

        # --- 4. Assemble Final Prompt ---
        system_message = textwrap.dedent(f"""\
            You are an expert driving model for an autonomous vehicle. Your task is to analyze a detailed driving scenario and predict the vehicle's next precise waypoint.
            Your response must be only the predicted (x, y) coordinates. Do not include any other text, reasoning, or formatting.
            
            Example Response Format: (1967.01, 7831.85)
            """)
        
        # Join the description lists into single strings before using them in the f-string.
        surrounding_vehicles_str = "\n".join(surrounding_vehicles_desc)
        road_layout_str = "\n".join(road_layout_desc)

        human_message = f"""\
        {self.delimiter} Driving Scenario Description {self.delimiter}
        Ego Vehicle State:
        {ego_state_desc}

        Surrounding Vehicles:
        {surrounding_vehicles_str}

        Road Layout:
        {road_layout_str}

        {self.delimiter} Your Task {self.delimiter}
        Driving Intention: {scene_data.get('Language Condition', 'Navigate safely and efficiently.')}
        Based on the complete scenario, predict the single most probable (x, y) coordinate for the ego vehicle's next waypoint in 0.5 seconds.
        """

        # In a real application, you would manage the system and human messages separately.
        # Here we combine them for a clear, single output.
        return f"---SYSTEM MESSAGE---\n{system_message}\n---HUMAN MESSAGE---\n{human_message}"

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

        agent = WaypointAgent()

        for time_idx in range(2,len(ego_traj['trajectory'])):
            # Generate the prompt for each time step
            final_prompt = agent.generate_waypoint_prompt(sample_data, current_time_idx=time_idx)

            print(f"Generated prompt for scenario {sid} with ego ID {ego_id} at time index {time_idx}:")
            print(final_prompt)
            print("-----------------------------------------------------")

        print(f"Converted data for scenario {sid} with ego ID {ego_id} to descriptor format.")
        breakpoint()

if __name__ == '__main__':
    filename = "/p/liverobotics/waymo_open_dataset_motion/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord-00000-of-00150"
    convert_data_language(filename)
