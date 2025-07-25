import math
import numpy as np
import torch
from typing import Union
from new_vector_utils import (
    VectorObservation,
    EgoField,
    RoadField,
    VehicleField,
    PedestrianField,
    angles_deg_and_distances,
    direction_to_angle_deg,
    VELOCITY_MS_SCALE,
    MS_TO_MPH,
)


def _transform_to_ego_frame(global_coords, ego_pos, ego_heading):
    """Helper function to transform global coordinates to the ego-relative frame."""
    translated_coords = global_coords - ego_pos
    cos_h = torch.cos(-ego_heading)
    sin_h = torch.sin(-ego_heading)
    rotated_x = translated_coords[:, 0] * cos_h - translated_coords[:, 1] * sin_h
    rotated_y = translated_coords[:, 0] * sin_h + translated_coords[:, 1] * cos_h
    return torch.stack([rotated_x, rotated_y], dim=1)

def xy_from_vehicle_desc(vehicle_array, ego_state):
    """Extracts XY coordinates from vehicles and converts them to the ego frame."""
    global_xy = torch.stack((vehicle_array[:, VehicleField.X], vehicle_array[:, VehicleField.Y]), dim=1)
    ego_pos = ego_state[EgoField.X:EgoField.Y+1]
    ego_heading = ego_state[EgoField.HEADING]
    valid_mask = vehicle_array[:, VehicleField.SPEED] > 1e-2
    relative_xy = torch.zeros_like(global_xy)
    if torch.any(valid_mask):
        relative_xy[valid_mask] = _transform_to_ego_frame(global_xy[valid_mask], ego_pos, ego_heading)
    return relative_xy

def xy_from_pedestrian_desc(pedestrian_array, ego_state):
    """Extracts XY coordinates from pedestrians and converts them to the ego frame."""
    global_xy = torch.stack((pedestrian_array[:, PedestrianField.X], pedestrian_array[:, PedestrianField.Y]), dim=1)
    ego_pos = ego_state[EgoField.X:EgoField.Y+1]
    ego_heading = ego_state[EgoField.HEADING]
    valid_mask = pedestrian_array[:, PedestrianField.SPEED] > 1e-2
    relative_xy = torch.zeros_like(global_xy)
    if torch.any(valid_mask):
        relative_xy[valid_mask] = _transform_to_ego_frame(global_xy[valid_mask], ego_pos, ego_heading)
    return relative_xy

def start_xy_from_road_desc(road_array, ego_state):
    """Extracts the starting XY of road features and converts them to the ego frame."""
    global_xy = torch.stack((road_array[:, RoadField.X1], road_array[:, RoadField.Y1]), dim=1)
    ego_pos = ego_state[EgoField.X:EgoField.Y+1]
    ego_heading = ego_state[EgoField.HEADING]
    return _transform_to_ego_frame(global_xy, ego_pos, ego_heading)

def flags_in_fov(relative_xy_coords, max_distance=40, fov_degrees=120):
    """Filters relative coordinates to find those within a forward-facing FOV."""
    distances, angular = angles_deg_and_distances(relative_xy_coords)
    return (
        (relative_xy_coords[:, 0] > 0)
        & (torch.abs(angular) < fov_degrees / 2)
        & (distances <= max_distance)
    )

def vehicle_filter_flags(vehicle_descriptors, ego_state):
    """Generates boolean flags for active vehicles within the field of view."""
    active_flags = vehicle_descriptors[:, VehicleField.SPEED] > 1e-2
    relative_xy = xy_from_vehicle_desc(vehicle_descriptors, ego_state)
    fov_flags = flags_in_fov(relative_xy, max_distance=80) 
    return active_flags & fov_flags

def pedestrian_filter_flags(pedestrian_descriptors, ego_state):
    """Generates boolean flags for active pedestrians within the field of view."""
    active_flags = pedestrian_descriptors[:, PedestrianField.SPEED] > 1e-2
    relative_xy = xy_from_pedestrian_desc(pedestrian_descriptors, ego_state)
    fov_flags = flags_in_fov(relative_xy, max_distance=30)
    return active_flags & fov_flags

# --- End of utility section ---

def get_road_feature_type(road_element_vector):
    """Identifies the type of a road feature from its one-hot encoded vector."""
    type_fields = [field for field in RoadField if field.name.startswith("TYPE_")]
    for field in type_fields:
        if road_element_vector[field.value] > 0.5:
            return field.name.replace("TYPE_", "").replace("_", " ").lower()
    return "unknown"


def make_waymo_observation_prompt(obs: Union[dict, VectorObservation], agent_id: bool = False) -> str:
    """
    Generates a human-readable text prompt describing the current driving scene
    based on the Waymo vector observation format, combining direct data and a narrative style.
    """
    if isinstance(obs, dict):
        obs = VectorObservation(**obs)

    # Convert numpy arrays to torch tensors if they are not already.
    if isinstance(obs.road_descriptors, np.ndarray):
        obs.road_descriptors = torch.from_numpy(obs.road_descriptors).float()
    if isinstance(obs.vehicle_descriptors, np.ndarray):
        obs.vehicle_descriptors = torch.from_numpy(obs.vehicle_descriptors).float()
    if isinstance(obs.pedestrian_descriptors, np.ndarray):
        obs.pedestrian_descriptors = torch.from_numpy(obs.pedestrian_descriptors).float()
    if isinstance(obs.ego_vehicle_descriptor, np.ndarray):
        obs.ego_vehicle_descriptor = torch.from_numpy(obs.ego_vehicle_descriptor).float()
    
    ego_state = obs.ego_vehicle_descriptor

    # 1. Describe Ego Vehicle State (Direct and Narrative)
    # ----------------------------------------------------
    current_speed_mph = ego_state[EgoField.SPEED] * VELOCITY_MS_SCALE * MS_TO_MPH
    
    if current_speed_mph > 55:
        speed_desc = f"I'm driving at highway speed, about {current_speed_mph:.0f} mph."
    elif current_speed_mph > 20:
        speed_desc = f"I'm driving at a moderate speed of {current_speed_mph:.0f} mph."
    elif current_speed_mph > 1:
        speed_desc = f"I'm driving slowly at {current_speed_mph:.0f} mph."
    else:
        speed_desc = "I am currently stopped."
        
    ego_direct_line = (
        f"Ego State: Speed={ego_state[EgoField.SPEED]:.2f}, "
        f"Position=({ego_state[EgoField.X]:.2f}, {ego_state[EgoField.Y]:.2f}), "
        f"Heading={ego_state[EgoField.HEADING]:.2f}"
    )

    # 2. Filter for and describe visible vehicles (Direct and Narrative)
    # ------------------------------------------------------------------
    vehicle_flags = vehicle_filter_flags(obs.vehicle_descriptors, ego_state)
    vehicles = obs.vehicle_descriptors[vehicle_flags]
    vehicle_lines = []
    if len(vehicles) > 0:
        relative_vehicle_xy = xy_from_vehicle_desc(vehicles, ego_state)
        distances, angles = angles_deg_and_distances(relative_vehicle_xy)
        sorted_indices = torch.argsort(distances)

        for i in sorted_indices:
            v_idx = i.item()
            v = vehicles[v_idx]
            dist, angle = distances[v_idx], angles[v_idx]
            v_speed_mph = v[VehicleField.SPEED] * VELOCITY_MS_SCALE * MS_TO_MPH
            
            # Direct Info
            v_id_str = f"Vehicle {v_idx}" if agent_id else "Vehicle"
            direct_info = (
                f"- {v_id_str} (raw): Speed={v[VehicleField.SPEED]:.2f}, "
                f"Position=({v[VehicleField.X]:.2f}, {v[VehicleField.Y]:.2f}), "
                f"Heading={v[VehicleField.HEADING]:.2f}"
            )
            
            # Narrative Info
            loc_desc = "ahead of me"
            if abs(angle) > 45: continue
            elif abs(angle) > 10 and angle < 0: loc_desc = "ahead and to my right"
            elif abs(angle) > 10 and angle > 0: loc_desc = "ahead and to my left"
            speed_desc_v = "is stationary." if v_speed_mph < 1 else f"is moving at roughly {v_speed_mph:.0f} mph."
            narrative_info = f"  (Narrative: A vehicle about {dist:.0f} meters {loc_desc} that {speed_desc_v})"
            
            vehicle_lines.append(direct_info + "\n" + narrative_info)


    # 3. Filter for and describe visible pedestrians (Direct and Narrative)
    # ---------------------------------------------------------------------
    ped_flags = pedestrian_filter_flags(obs.pedestrian_descriptors, ego_state)
    pedestrians = obs.pedestrian_descriptors[ped_flags]
    pedestrian_lines = []
    if len(pedestrians) > 0:
        relative_ped_xy = xy_from_pedestrian_desc(pedestrians, ego_state)
        distances, angles = angles_deg_and_distances(relative_ped_xy)
        sorted_indices = torch.argsort(distances)
        for i in sorted_indices:
            p_idx = i.item()
            p = pedestrians[p_idx]
            dist, angle = distances[p_idx], angles[p_idx]

            # Direct Info
            p_id_str = f"Pedestrian {p_idx}" if agent_id else "Pedestrian"
            direct_info = (
                f"- {p_id_str} (raw): Speed={p[PedestrianField.SPEED]:.2f}, "
                f"Position=({p[PedestrianField.X]:.2f}, {p[PedestrianField.Y]:.2f}), "
                f"Heading={p[PedestrianField.HEADING]:.2f}"
            )
            
            # Narrative Info
            loc_desc = "ahead"
            if angle < -10: loc_desc = "to my right"
            elif angle > 10: loc_desc = "to my left"
            narrative_info = f"  (Narrative: I see a pedestrian {dist:.0f} meters away {loc_desc}.)"
            
            pedestrian_lines.append(direct_info + "\n" + narrative_info)

    # 4. Identify and summarize road features (Direct and Narrative)
    # --------------------------------------------------------------
    type_flags_start = RoadField.TYPE_LANE_CENTER_FREEWAY
    type_flags_end = RoadField.TYPE_SPEED_BUMP + 1
    active_road_flags = torch.sum(obs.road_descriptors[:, type_flags_start:type_flags_end], dim=1) > 0.5
    road_elements = obs.road_descriptors[active_road_flags]
    
    road_feature_lines = []
    salient_features = []
    lane_center_lines = []
    if len(road_elements) > 0:
        for elem in road_elements:
            feature_type = get_road_feature_type(elem)
            road_feature_lines.append(
                f"- Road Feature: Type={feature_type}, "
                f"StartPosition=({elem[RoadField.X1]:.2f}, {elem[RoadField.Y1]:.2f})"
            )
            if "lane center" in feature_type:
                lane_center_lines.append(elem)
            elif "stop sign" in feature_type or "crosswalk" in feature_type or "speed bump" in feature_type:
                relative_start_point = _transform_to_ego_frame(elem[RoadField.X1:RoadField.Y1+1].unsqueeze(0), ego_state[EgoField.X:EgoField.Y+1], ego_state[EgoField.HEADING])
                dist = torch.linalg.norm(relative_start_point)
                if dist < 100:
                    salient_features.append(f"There is a {feature_type} about {dist:.0f}m ahead.")

    # 5. Describe the upcoming path using lane centerlines
    # ----------------------------------------------------
    path_line = "I need to continue forward."
    if lane_center_lines:
        closest_lane_dist = float('inf')
        lane_to_follow = None
        for line in lane_center_lines:
            relative_start_point = _transform_to_ego_frame(line[RoadField.X1:RoadField.Y1+1].unsqueeze(0), ego_state[EgoField.X:EgoField.Y+1], ego_state[EgoField.HEADING])
            dist = torch.linalg.norm(relative_start_point)
            if dist < closest_lane_dist:
                closest_lane_dist = dist
                lane_to_follow = line

        if lane_to_follow is not None:
            last_valid_idx = -1
            for i in range(1, 11):
                if lane_to_follow[getattr(RoadField, f"INVALID_{i}")] < 0.5:
                    last_valid_idx = i
                else: break

            if last_valid_idx != -1:
                end_x_global = lane_to_follow[getattr(RoadField, f"X{last_valid_idx}")]
                end_y_global = lane_to_follow[getattr(RoadField, f"Y{last_valid_idx}")]
                relative_end_point = _transform_to_ego_frame(torch.tensor([[end_x_global, end_y_global]]), ego_state[EgoField.X:EgoField.Y+1], ego_state[EgoField.HEADING])
                dist_to_end = torch.linalg.norm(relative_end_point)
                angle_to_end = direction_to_angle_deg(relative_end_point[:, 0], relative_end_point[:, 1]).item()

                if dist_to_end > 10:
                    if abs(angle_to_end) < 15:
                        path_line = f"The road ahead is straight for at least {dist_to_end:.0f} meters."
                    elif angle_to_end > 15:
                        path_line = f"The road ahead curves to the left."
                    else:
                        path_line = f"The road ahead curves to the right."

    # 6. Assemble the final prompt
    # ----------------------------
    summary = f"I see {len(vehicles)} vehicle(s) and {len(pedestrians)} pedestrian(s)."
    
    prompt_sections = ["## My Status ##", speed_desc, ego_direct_line]
    
    if vehicle_lines:
        prompt_sections.append("\n## Nearby Vehicles ##")
        prompt_sections.extend(vehicle_lines)
    if pedestrian_lines:
        prompt_sections.append("\n## Nearby Pedestrians ##")
        prompt_sections.extend(pedestrian_lines)
    if road_feature_lines:
        prompt_sections.append("\n## Road Infrastructure (Raw Data) ##")
        prompt_sections.extend(road_feature_lines)
    if salient_features:
         prompt_sections.append("\n## Important Features Ahead ##")
         prompt_sections.extend(salient_features)
         
    prompt_sections.append(f"\n## Path Summary ##\n{path_line}")

    return "\n".join(prompt_sections)

if __name__ == '__main__':
    # --- Create a sample data dictionary for testing ---
    sample_data = {
        'Map Data': [{'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1963.56982422, 7806.85546875]), np.array([1963.52709961, 7843.18310547])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1966.96679688, 7806.84423828]), np.array([1966.98547363, 7843.69970703])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1966.93896484, 7785.59375]), np.array([1966.95495605, 7793.99853516]), np.array([1969.23913574, 7797.90966797]), np.array([1970.70812988, 7798.00439453])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1966.94824219, 7793.01367188]), np.array([1966.96679688, 7806.84423828])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1963.64428711, 7792.94677734]), np.array([1963.56982422, 7806.85546875])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1970.98742676, 7801.27636719]), np.array([1969.01513672, 7801.32470703]), np.array([1967.06616211, 7801.60839844]), np.array([1963.73632812, 7804.89355469]), np.array([1963.56982422, 7806.85546875])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1970.98742676, 7801.27636719]), np.array([1962.15856934, 7801.75683594])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1967.05712891, 7804.48193359]), np.array([1966.96679688, 7806.84423828])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1970.98742676, 7801.27636719]), np.array([1967.02539062, 7801.43017578]), np.array([1962.11303711, 7802.08984375])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1970.98742676, 7801.27636719]), np.array([1968.0501709, 7801.18896484]), np.array([1962.22180176, 7800.45507812])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1963.92077637, 7759.53564453]), np.array([1966.51672363, 7776.18798828]), np.array([1966.83496094, 7780.14111328]), np.array([1966.93896484, 7785.59375])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1963.57446289, 7760.57958984]), np.array([1963.64428711, 7792.94677734])]}, {'type': 'RoadLine-BrokenSingleWhite', 'pos_xy': [np.array([1961.87744141, 7761.74658203]), np.array([1961.93127441, 7802.21337891])]}, {'type': 'RoadLine-SolidSingleWhite', 'pos_xy': [np.array([1965.40332031, 7785.50488281]), np.array([1965.16125488, 7843.51123047])]}, {'type': 'RoadEdgeBoundary', 'pos_xy': [np.array([1968.43725586, 7822.7265625]), np.array([1968.78320312, 7845.12158203])]}, {'type': 'RoadEdgeBoundary', 'pos_xy': [np.array([1968.4642334, 7806.51367188]), np.array([1968.43725586, 7822.7265625])]}, {'type': 'RoadEdgeBoundary', 'pos_xy': [np.array([1967.46484375, 7760.70117188]), np.array([1968.63195801, 7782.35400391]), np.array([1968.51171875, 7794.00048828]), np.array([1971.43725586, 7794.03027344])]}, {'type': 'Crosswalk', 'pos_xy': [np.array([1965.60144043, 7845.03515625]), np.array([1968.67199707, 7845.16992188])]}],
        'Ego Trajectory': {'type': 'Vehicle', 'trajectory': np.array([[-1.00000000e+00, -1.00000000e+00], [1.96526404e+03, 7.76571191e+03], [1.96557971e+03, 7.77292139e+03], [1.96608276e+03, 7.78193604e+03], [1.96643823e+03, 7.78937842e+03], [1.96669189e+03, 7.79597900e+03], [1.96683398e+03, 7.80195312e+03], [1.96690027e+03, 7.80714648e+03], [1.96693176e+03, 7.81174316e+03], [1.96695154e+03, 7.81586426e+03], [1.96697522e+03, 7.81946924e+03], [1.96698767e+03, 7.82249658e+03], [1.96699011e+03, 7.82487500e+03], [1.96699292e+03, 7.82656885e+03], [1.96699207e+03, 7.82779785e+03], [1.96698657e+03, 7.82872119e+03], [1.96698218e+03, 7.82952588e+03], [1.96697607e+03, 7.83041113e+03], [1.96697656e+03, 7.83131445e+03]])},
        'Nearby Agent Trajectories': {650: {'type': 'Vehicle', 'trajectory': np.array([[1966.87072754, 7828.43798828], [1966.87475586, 7828.45458984], [1966.86901855, 7828.484375], [1966.86706543, 7828.66162109], [1966.86401367, 7829.0703125], [1966.85656738, 7829.80566406], [1966.84875488, 7830.76269531], [1966.83544922, 7831.95849609], [1966.8236084, 7833.11181641], [1966.81616211, 7834.24609375], [1966.81835938, 7835.26904297], [1966.81335449, 7836.15917969], [1966.80432129, 7836.77832031], [1966.7989502, 7837.21044922], [1966.81237793, 7837.69873047], [1966.82092285, 7838.26416016], [1966.83056641, 7838.88818359], [1966.83239746, 7839.64648438], [1966.84216309, 7840.64794922]])}, 672: {'type': 'Vehicle', 'trajectory': np.array([[1964.07141113, 7790.80615234], [1964.06347656, 7795.94921875], [1964.04870605, 7800.78564453], [1964.0435791, 7805.06054688], [1964.01977539, 7809.04199219], [1964.00061035, 7812.54638672], [1963.9921875, 7815.69042969], [1963.96240234, 7818.52832031], [1963.95715332, 7821.07714844], [1963.95263672, 7823.296875], [1963.94311523, 7825.2109375], [1963.94128418, 7826.83349609], [1963.94030762, 7828.14355469], [1963.94372559, 7829.20605469], [1963.94592285, 7830.00097656], [1963.94299316, 7830.61865234], [1963.94763184, 7831.14111328], [1963.94812012, 7831.54931641], [1963.9543457, 7831.82324219]])}}
    }

    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    from data_conversion.convert_data_utils import convert_to_descriptor_format


    converted_data = convert_to_descriptor_format(sample_data)

    converted_data = converted_data[2]  # get the third time step for demonstration, which is the 10th second of original waymo motion

    np.set_printoptions(threshold=sys.maxsize)

    print("Converted Data:")
    print(converted_data)
    breakpoint()  
    prompt = make_waymo_observation_prompt(converted_data)

    print(prompt)