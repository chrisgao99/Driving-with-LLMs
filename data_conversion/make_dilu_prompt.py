import math
import numpy as np
import textwrap
import openai
import re
import os
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
        # It's better to get the API key from an environment variable,
        # but this will use the hardcoded key as requested.
        self.api_key = "api key here"
        self.client = openai.OpenAI(api_key=self.api_key)


    def _calculate_dynamics(self, trajectory: np.ndarray, current_time_idx: int, dt: float = 0.5):
        """Calculates the dynamic state (speed, heading, acceleration) for an agent."""
        if current_time_idx < 1 or np.all(trajectory[current_time_idx] == -1) or np.all(trajectory[current_time_idx - 1] == -1):
            return 0.0, 0.0, 0.0
        p_curr, p_prev = trajectory[current_time_idx], trajectory[current_time_idx - 1]
        dx, dy = p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]
        distance = math.sqrt(dx**2 + dy**2)
        speed = distance / dt
        heading_deg = math.degrees(math.atan2(dy, dx))
        acceleration = 0.0
        if current_time_idx >= 2 and not np.all(trajectory[current_time_idx - 2] == -1):
            p_prev2 = trajectory[current_time_idx - 2]
            dx_prev, dy_prev = p_prev[0] - p_prev2[0], p_prev[1] - p_prev2[1]
            speed_prev = math.sqrt(dx_prev**2 + dy_prev**2) / dt
            acceleration = (speed - speed_prev) / dt
        return speed, heading_deg, acceleration

    def generate_waypoint_prompt(self, scene_data: dict, current_time_idx: int) -> str:
        """Constructs a detailed prompt for an LLM to predict the next waypoint."""
        ego_traj = scene_data['Ego Trajectory']['trajectory']
        ego_x, ego_y = ego_traj[current_time_idx]
        ego_speed, ego_heading, ego_accel = self._calculate_dynamics(ego_traj, current_time_idx)
        ego_state_desc = textwrap.dedent(f"""\
            - Your current position (x, y) is ({ego_x:.2f}, {ego_y:.2f}).
            - Your speed is {ego_speed:.2f} m/s.
            - Your heading is {ego_heading:.2f} degrees.
            - Your acceleration is {ego_accel:.2f} m/s^2.""")

        surrounding_vehicles_desc = []
        for agent_id, agent_data in scene_data.get('Nearby Agent Trajectories', {}).items():
            agent_traj = agent_data['trajectory']
            if current_time_idx >= len(agent_traj) or np.all(agent_traj[current_time_idx] == -1):
                continue
            agent_x, agent_y = agent_traj[current_time_idx]
            distance_to_ego = math.sqrt((ego_x - agent_x)**2 + (ego_y - agent_y)**2)
            if distance_to_ego > 100:
                continue
            speed, heading, _ = self._calculate_dynamics(agent_traj, current_time_idx)
            surrounding_vehicles_desc.append(
                f"- Vehicle `{agent_id}` is {distance_to_ego:.2f} meters away at position ({agent_x:.2f}, {agent_y:.2f}).\n"
                f"  It is traveling at {speed:.2f} m/s with a heading of {heading:.2f} degrees."
            )
        if not surrounding_vehicles_desc:
            surrounding_vehicles_desc.append("No other vehicles are nearby.")

        road_layout_desc = []
        for road_segment in scene_data.get('Map Data', []):
            points = np.array(road_segment['pos_xy'])
            if points.size == 0: continue
            min_dist = np.min(np.linalg.norm(points - np.array([ego_x, ego_y]), axis=1))
            if min_dist < 20:
                road_type = road_segment['type'].replace('-', ' ').replace('RoadLine', 'Road Line')
                points_str = ", ".join([f"({p[0]:.2f}, {p[1]:.2f})" for p in points])
                road_layout_desc.append(f"- A '{road_type}' is nearby with the following points:\n  Coordinates: [{points_str}]")
        if not road_layout_desc:
            road_layout_desc.append("No relevant map features are nearby.")

        system_message = textwrap.dedent(f"""\
            You are an expert driving model for an autonomous vehicle. Your task is to calculate the vehicle's next precise waypoint.
            Your response must be only the predicted (x, y) coordinates. Do not include any other text, reasoning, or formatting.
            Example Response Format: (1967.01, 7831.85)""")
        
        surrounding_vehicles_str = "\n".join(surrounding_vehicles_desc)
        road_layout_str = "\n".join(road_layout_desc)
        driving_intention = scene_data.get('Language Condition', 'Navigate safely and efficiently.')
        
        human_message = f"""\
        {self.delimiter} Driving Scenario Description {self.delimiter}
        Ego Vehicle State:
        {ego_state_desc}
        Surrounding Vehicles:
        {surrounding_vehicles_str}
        Road Layout:
        {road_layout_str}
        {self.delimiter} Your Task {self.delimiter}
        Driving Intention: {driving_intention}
        Based on the complete scenario, predict the single most probable (x, y) coordinate for the ego vehicle's next waypoint in 0.5 seconds."""
        
        return f"---SYSTEM MESSAGE---\n{system_message}\n\n---HUMAN MESSAGE---\n{human_message}"

    def query_gpt_for_waypoint(self, prompt: str):
        """
        Queries the GPT model with the provided prompt and parses the response
        to extract the next waypoint.
        """
        if not self.client.api_key:
            print("Warning: OpenAI API key not set. Skipping GPT query.")
            return None

        try:
            parts = prompt.split("---HUMAN MESSAGE---")
            system_prompt = parts[0].replace("---SYSTEM MESSAGE---", "").strip()
            human_prompt = parts[1].strip()

            response = self.client.chat.completions.create(
                model="gpt-4",  # Or "gpt-3.5-turbo"
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": human_prompt}
                ],
                temperature=0.0,
                max_tokens=25
            )
            content = response.choices[0].message.content.strip()

            # Use regex to find a pattern like (123.45, 678.90)
            match = re.search(r'\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)', content)
            if match:
                x = float(match.group(1))
                y = float(match.group(2))
                return (x, y)
            else:
                print(f"Warning: Could not parse waypoint from GPT response: '{content}'")
                return None
        except Exception as e:
            print(f"An error occurred while querying GPT: {e}")
            return None

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


def querry_gpt_dilu(tfrecord_path: str):
    """
    Processes a TFRecord file, iterates through scenarios, and queries the LLM
    for a waypoint at each valid timestep using auto-regressive prediction.
    """
    data = run_vectorize_process(tfrecord_path, "language_condition")

    map_dict = data["map_dict"]
    tf_cleaned_traj_dict = data["tf_cleaned_traj_dict"]
    language_condition_data = data["language_condition_data"]
    valid_indices = data["valid_indices"]

    # Instantiate the agent once before the loop for efficiency
    agent = WaypointAgent()

    for sid_egoid, road_segments_list in map_dict.items():
        parts = sid_egoid.split("__")
        sid, ego_id = parts[0], int(parts[1])
        print(f"\n--- Processing Scenario: {sid} | Ego ID: {ego_id} ---")

        ego_traj_data = tf_cleaned_traj_dict[sid][ego_id]
        
        other_agent_trajs = {
            agent_id: tf_cleaned_traj_dict[sid][agent_id]
            for agent_id in valid_indices[sid][ego_id]['valid_agent']
            if agent_id in tf_cleaned_traj_dict[sid]
        }

        # Create a dynamic copy of the ego trajectory to update with predictions
        dynamic_ego_trajectory = np.copy(ego_traj_data['trajectory'])
        dynamic_ego_trajectory = _interpolate_invalid_points(dynamic_ego_trajectory)
    
        # nearby_agents = data.get('Nearby Agent Trajectories', {})
        processed_nearby_agents = {}
        for agent_id, agent_data in other_agent_trajs.items():
            processed_data = agent_data.copy()
            processed_data['trajectory'] = _interpolate_invalid_points(agent_data['trajectory'])
            processed_nearby_agents[agent_id] = processed_data

        sample_data = {
            'Map Data': road_segments_list,
            'Ego Trajectory': {'trajectory': dynamic_ego_trajectory}, # Use the dynamic trajectory
            'Nearby Agent Trajectories': processed_nearby_agents,
            'Language Condition': language_condition_data.get(sid_egoid)
        }
        
        # List to store the predicted waypoints for the current scenario
        predicted_trajectory_points = []
        
        # The first few points of the trajectory are from ground truth
        for i in range(2):
            predicted_trajectory_points.append(dynamic_ego_trajectory[i])

        # Loop through each valid time step for the current scenario
        for time_idx in range(2, len(ego_traj_data['trajectory'])):
            print(f"\n-- Predicting for Timestep: {time_idx} --")
            
            # 1. Generate the prompt using the current state of the dynamic trajectory
            final_prompt = agent.generate_waypoint_prompt(sample_data, current_time_idx=time_idx)
            print(f"Generated Prompt:\n{final_prompt}\n")
            
            # 2. Query GPT with the prompt
            predicted_waypoint = agent.query_gpt_for_waypoint(final_prompt)
            
            if predicted_waypoint:
                print(f"--> GPT Predicted Waypoint: {predicted_waypoint}")
                predicted_trajectory_points.append(predicted_waypoint)
                
                # IMPORTANT: Update the *next* position in the dynamic trajectory
                # with the waypoint we just predicted.
                if time_idx + 1 < len(dynamic_ego_trajectory):
                    dynamic_ego_trajectory[time_idx + 1] = predicted_waypoint
            else:
                print("--> Failed to get waypoint from GPT. Stopping prediction for this scenario.")
                break # Exit the loop for this scenario if a prediction fails

        # After the loop, concatenate the waypoints into a single NumPy array
        if len(predicted_trajectory_points) > 2: # Check if any predictions were made
            full_predicted_trajectory = np.array(predicted_trajectory_points)
            print(f"\n--- Full Predicted Trajectory for Scenario {sid_egoid} ---")
            print(full_predicted_trajectory)
            print(f"Shape: {full_predicted_trajectory.shape}")
        else:
            print(f"\nNo waypoints were successfully predicted for scenario {sid_egoid}.")

        print("groud truth trajectory: ", ego_traj_data['trajectory'])
        print(f"Finished processing scenario {sid} with ego ID {ego_id}.")
        breakpoint()

if __name__ == '__main__':
    # This path should point to your actual TFRecord file.
    filename = "/p/liverobotics/waymo_open_dataset_motion/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord-00000-of-00150"
    
    # Check if the file exists before running
    if os.path.exists(filename):
         querry_gpt_dilu(filename)
    else:
        print(f"Error: File not found at '{filename}'.")
        print("Please update the 'filename' variable with the correct path to your TFRecord file.")
