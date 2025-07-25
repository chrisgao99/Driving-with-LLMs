import numpy as np
import torch
from vector_utils import (
    VectorObservation,
    EgoField,
    RoadField,
    vehicle_filter_flags,
    pedestrian_filter_flags,
    xy_from_vehicle_desc,
    xy_from_pedestrian_desc,
    start_xy_from_road_desc,
    traveling_angle_deg_from_vehicle_desc,
    traveling_angle_deg_from_pedestrian_desc,
    angles_deg_and_distances,
    direction_to_angle_deg,
    VELOCITY_MS_SCALE,
    MS_TO_MPH,
)

def object_direction(angle_deg):
    """Converts a directional angle into a human-readable string."""
    if abs(angle_deg) < 45:
        return "traveling in the same direction as me"
    elif abs(abs(angle_deg) - 180) < 45:
        return "traveling in the opposite direction from me"
    elif angle_deg > 0: # Moving left to right relative to the ego car's heading
        return "moving from my left to my right"
    else: # Moving right to left
        return "moving from my right to my left"

def get_road_feature_type(road_element_vector):
    """Identifies the type of a road feature from its one-hot encoded vector."""
    type_fields = [field for field in RoadField if field.name.startswith("TYPE_")]
    for field in type_fields:
        if road_element_vector[field.value] > 0.5:
            # Cleans up the name for readability
            return field.name.replace("TYPE_", "").replace("_", " ").lower()
    return "unknown road feature"


def make_waymo_observation_prompt(obs: VectorObservation, agent_id: bool = False) -> str:
    """
    Generates a human-readable text prompt describing the current driving scene
    based on the Waymo vector observation format.
    """
    if isinstance(obs, dict):
        obs = VectorObservation(**obs)

    # 1. Filter for and describe visible vehicles
    # --------------------------------------------
    vehicle_flags = vehicle_filter_flags(obs.vehicle_descriptors)
    vehicles = obs.vehicle_descriptors[vehicle_flags]
    vehicle_lines = []
    if len(vehicles) > 0:
        distances_vehicles, angular_vehicles = angles_deg_and_distances(xy_from_vehicle_desc(vehicles))
        vehicle_traveling_direction = traveling_angle_deg_from_vehicle_desc(vehicles)
        for i in range(len(vehicles)):
            v_id_str = f" (id={i})" if agent_id else ""
            vehicle_lines.append(
                f"- A vehicle{v_id_str} is {distances_vehicles[i]:.1f}m away at a {angular_vehicles[i]:.1f}-degree angle, {object_direction(vehicle_traveling_direction[i])}."
            )

    # 2. Filter for and describe visible pedestrians
    # ----------------------------------------------
    ped_flags = pedestrian_filter_flags(obs.pedestrian_descriptors)
    pedestrians = obs.pedestrian_descriptors[ped_flags]
    pedestrian_lines = []
    if len(pedestrians) > 0:
        distances_peds, angular_peds = angles_deg_and_distances(xy_from_pedestrian_desc(pedestrians))
        pedestrian_traveling_direction = traveling_angle_deg_from_pedestrian_desc(pedestrians)
        for i in range(len(pedestrians)):
            p_id_str = f" (id={i})" if agent_id else ""
            pedestrian_lines.append(
                f"- A pedestrian{p_id_str} is {distances_peds[i]:.1f}m away at a {angular_peds[i]:.1f}-degree angle, {object_direction(pedestrian_traveling_direction[i])}."
            )

    # 3. Describe Ego Vehicle State
    # -------------------------------
    current_speed_mph = obs.ego_vehicle_descriptor[EgoField.SPEED] * VELOCITY_MS_SCALE * MS_TO_MPH
    ego_heading_deg = obs.ego_vehicle_descriptor[EgoField.HEADING] * 180 / math.pi
    ego_state_line = f"My vehicle is currently traveling at {current_speed_mph:.1f} mph with a heading of {ego_heading_deg:.1f} degrees."

    # 4. Identify and describe nearby static road features
    # ----------------------------------------------------
    # A road feature is active if its first point is not marked as invalid
    active_road_flags = obs.road_descriptors[:, RoadField.INVALID_1] < 0.5
    road_elements = obs.road_descriptors[active_road_flags]
    road_feature_lines = []
    lane_center_lines = []
    if len(road_elements) > 0:
        road_xy = start_xy_from_road_desc(road_elements)
        distances_road, angular_road = angles_deg_and_distances(road_xy)
        for i in range(len(road_elements)):
            feature_type = get_road_feature_type(road_elements[i])
            # Only describe salient, non-lane features here
            if "lane" not in feature_type and "road edge" not in feature_type:
                 road_feature_lines.append(
                    f"- A {feature_type} is {distances_road[i]:.1f}m away at a {angular_road[i]:.1f}-degree angle."
                )
            elif "lane center" in feature_type:
                 lane_center_lines.append(road_elements[i])


    # 5. Describe the upcoming path using lane centerlines
    # ----------------------------------------------------
    path_line = "The path ahead is unclear."
    if lane_center_lines:
        # Find the closest lane centerline to follow
        closest_lane_dist = float('inf')
        lane_to_follow = None
        for line in lane_center_lines:
            dist = np.linalg.norm([line[RoadField.X1], line[RoadField.Y1]])
            if dist < closest_lane_dist:
                closest_lane_dist = dist
                lane_to_follow = line

        if lane_to_follow is not None:
            # Find the last valid point of the lane to describe the path
            last_valid_idx = -1
            for i in range(1, 11):
                if lane_to_follow[getattr(RoadField, f"INVALID_{i}")] < 0.5:
                    last_valid_idx = i
                else:
                    break # Stop at the first invalid point

            if last_valid_idx != -1:
                end_x = lane_to_follow[getattr(RoadField, f"X{last_valid_idx}")]
                end_y = lane_to_follow[getattr(RoadField, f"Y{last_valid_idx}")]
                dist_to_end = np.linalg.norm([end_x, end_y])
                angle_to_end = direction_to_angle_deg(torch.tensor([end_x]), torch.tensor([end_y])).item()

                if abs(angle_to_end) < 15:
                    path_line = f"The lane ahead continues straight for at least {dist_to_end:.0f}m."
                elif angle_to_end > 15:
                    path_line = f"The lane ahead curves to the left in {dist_to_end:.0f}m."
                else:
                    path_line = f"The lane ahead curves to the right in {dist_to_end:.0f}m."

    # 6. Assemble the final prompt
    # ----------------------------
    prompt_sections = [
        f"Scene Description:",
        ego_state_line,
        f"\nI see {len(vehicles)} vehicle(s) and {len(pedestrians)} pedestrian(s) nearby.",
    ]
    if vehicle_lines:
        prompt_sections.append("\nVehicles:\n" + "\n".join(vehicle_lines))
    if pedestrian_lines:
        prompt_sections.append("\nPedestrians:\n" + "\n".join(pedestrian_lines))
    if road_feature_lines:
        prompt_sections.append("\nRoad Features:\n" + "\n".join(road_feature_lines))
    prompt_sections.append(f"\nPath ahead: {path_line}")

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

    # Get the absolute path of the directory containing the current script
    # __file__ is the pathname of the file from which the module was loaded.
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the parent directory path by going one level up
    parent_dir = os.path.dirname(current_dir)

    # Add the parent directory to the system path
    sys.path.append(parent_dir)

    # Now you can import the module as if it were in the same directory
    from utils.convert_data_utils import convert_to_descriptor_format


    converted_data = convert_to_descriptor_format(sample_data)

    converted_data = converted_data[2]  # get the third time step for demonstration, which is the 10th second of original waymo motion

    # --- Generate and print the QA pairs ---
    generated_qa = generate_qa_for_timestep(converted_data)
    
    print("--- Generated QA Pairs ---")
    for i, qa in enumerate(generated_qa):
        print(f"\n{i+1}. Question: {qa['question']}")
        print(f"   Answer: {qa['answer']}")