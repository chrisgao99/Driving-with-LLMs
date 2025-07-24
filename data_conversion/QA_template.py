import numpy as np
from enum import IntEnum
from vector_utils_waymo_custom import VehicleField, PedestrianField, RoadField, EgoField 

# Helper to get road type name from index
ROAD_TYPE_NAMES = {v: k.replace('TYPE_', '').replace('_', ' ').title() for k, v in RoadField.__members__.items() if 'TYPE' in k}

def generate_qa_for_timestep(data):
    """
    Generates a list of 20 QA pairs based on a single timestep of structured data.

    Args:
        data (dict): A dictionary containing the descriptor arrays for one timestep.
                     Keys: 'ego_vehicle_descriptor', 'vehicle_descriptors',
                           'pedestrian_descriptors', 'road_descriptors'.

    Returns:
        list: A list of 20 dictionaries, each with a "question" and "answer" key.
    """
    qa_pairs = []
    
    # Unpack data for easier access
    ego = data['ego_vehicle_descriptor']
    vehicles = data['vehicle_descriptors']
    pedestrians = data['pedestrian_descriptors']
    roads = data['road_descriptors']

    # --- Question Generation ---
    
    # 1. Ego Acceleration
    qa_pairs.append({
        "question": "What is the ego vehicle's current acceleration?",
        "answer": f"The ego vehicle's acceleration is {ego[EgoField.ACCEL]:.2f} m/s^2."
    })

    # 2. Ego Speed
    qa_pairs.append({
        "question": "What is the current speed of the ego vehicle?",
        "answer": f"The ego vehicle's speed is {ego[EgoField.SPEED]:.2f} m/s."
    })

    # 3. Ego Coordinates
    qa_pairs.append({
        "question": "What are the current (x, y) coordinates of the ego vehicle?",
        "answer": f"The ego vehicle is at coordinates ({ego[EgoField.X]:.2f}, {ego[EgoField.Y]:.2f})."
    })

    # 4. Ego Velocity
    qa_pairs.append({
        "question": "What is the ego vehicle's velocity vector?",
        "answer": f"The ego vehicle's velocity is (dx={ego[EgoField.DX]:.2f}, dy={ego[EgoField.DY]:.2f})."
    })

    # 5. Nearby Vehicles
    num_vehicles = np.count_nonzero(np.any(vehicles, axis=1))
    if num_vehicles > 0:
        answer = f"Yes, there are {num_vehicles} other vehicles nearby. The first one is at coordinates ({vehicles[0, VehicleField.X]:.2f}, {vehicles[0, VehicleField.Y]:.2f})."
    else:
        answer = "No, there are no other vehicles nearby."
    qa_pairs.append({
        "question": "Are there any other vehicles nearby? If so, what are the coordinates of the first one?",
        "answer": answer
    })

    # 6. Nearby Vehicle Speed
    qa_pairs.append({
        "question": "What is the speed of the first nearby vehicle?",
        "answer": f"The speed of the first nearby vehicle is {vehicles[0, VehicleField.SPEED]:.2f} m/s." if num_vehicles > 0 else "There are no nearby vehicles."
    })

    # 7. Nearby Vehicle Velocity
    qa_pairs.append({
        "question": "What is the velocity of the first nearby vehicle?",
        "answer": f"The velocity of the first nearby vehicle is (dx={vehicles[0, VehicleField.DX]:.2f}, dy={vehicles[0, VehicleField.DY]:.2f})." if num_vehicles > 0 else "There are no nearby vehicles."
    })

    # 8. Pedestrians
    num_peds = np.count_nonzero(np.any(pedestrians, axis=1))
    if num_peds > 0:
        answer = f"Yes, there are {num_peds} pedestrians. The first one is at coordinates ({pedestrians[0, PedestrianField.X]:.2f}, {pedestrians[0, PedestrianField.Y]:.2f})."
    else:
        answer = "No, there are no pedestrians in the scene."
    qa_pairs.append({
        "question": "Are there any pedestrians in the scene? If so, what are the coordinates of the first one listed?",
        "answer": answer
    })

    # 9. Pedestrian Speed
    qa_pairs.append({
        "question": "What is the speed of the first pedestrian?",
        "answer": f"The speed of the first pedestrian is {pedestrians[0, PedestrianField.SPEED]:.2f} m/s." if num_peds > 0 else "There are no pedestrians."
    })

    # 10. Pedestrian Velocity
    qa_pairs.append({
        "question": "What is the velocity of the first pedestrian?",
        "answer": f"The velocity of the first pedestrian is (dx={pedestrians[0, PedestrianField.DX]:.2f}, dy={pedestrians[0, PedestrianField.DY]:.2f})." if num_peds > 0 else "There are no pedestrians."
    })

    # 11. All Road Types
    road_types = set()
    for i in range(len(roads)):
        if np.any(roads[i]): # Check if the row is not all zeros
            type_idx = np.argmax(roads[i, 30:]) + 30
            if roads[i, type_idx] == 1.0:
                road_types.add(ROAD_TYPE_NAMES.get(type_idx, "Unknown"))
    qa_pairs.append({
        "question": "What are all the road segment types present in the current view?",
        "answer": f"The following road types are present: {', '.join(sorted(list(road_types)))}." if road_types else "There are no road segments with defined types."
    })

    # 12. Road Point 1
    qa_pairs.append({
        "question": "What are the coordinates of the first point of the first road segment?",
        "answer": f"The coordinates are ({roads[0, RoadField.X1]:.2f}, {roads[0, RoadField.Y1]:.2f})." if np.any(roads) else "There are no road segments."
    })

    # 13. Road Point 5
    qa_pairs.append({
        "question": "What are the coordinates of the fifth point of the first road segment?",
        "answer": f"The coordinates are ({roads[0, RoadField.X5]:.2f}, {roads[0, RoadField.Y5]:.2f})." if np.any(roads) else "There are no road segments."
    })

    # 14. Road Point 10
    qa_pairs.append({
        "question": "What are the coordinates of the tenth point of the first road segment?",
        "answer": f"The coordinates are ({roads[0, RoadField.X10]:.2f}, {roads[0, RoadField.Y10]:.2f})." if np.any(roads) else "There are no road segments."
    })

    # 15. Surface Street
    is_surface_street = np.any(roads[:, RoadField.TYPE_LANE_CENTER_SURFACE_STREET] == 1.0)
    qa_pairs.append({
        "question": "Are there any surface street lanes in the data?",
        "answer": "Yes, at least one road segment is a surface street lane." if is_surface_street else "No, there are no surface street lanes."
    })

    # 16. Crosswalk
    is_crosswalk = np.any(roads[:, RoadField.TYPE_CROSSWALK] == 1.0)
    qa_pairs.append({
        "question": "Is a crosswalk identified in the road data?",
        "answer": "Yes, a crosswalk is present in the data." if is_crosswalk else "No, there are no crosswalks."
    })

    # 17. Solid White Line
    is_solid_white = np.any(roads[:, RoadField.TYPE_ROAD_LINE_SOLID_SINGLE_WHITE] == 1.0)
    qa_pairs.append({
        "question": "Is there a solid single white line in the data?",
        "answer": "Yes, at least one road segment is a solid single white line." if is_solid_white else "No, there are no solid single white lines."
    })

    # 18. Freeway Lane
    is_freeway = np.any(roads[:, RoadField.TYPE_LANE_CENTER_FREEWAY] == 1.0)
    qa_pairs.append({
        "question": "Is a freeway lane represented in the road data?",
        "answer": "Yes, a freeway lane is present in the data." if is_freeway else "No, there are no freeway lanes."
    })

    # 19. Total Nearby Vehicles
    qa_pairs.append({
        "question": "How many nearby vehicles are there in total?",
        "answer": f"There are a total of {num_vehicles} nearby vehicles."
    })

    # 20. Total Pedestrians
    qa_pairs.append({
        "question": "How many pedestrians are there in total?",
        "answer": f"There are a total of {num_peds} pedestrians."
    })

    return qa_pairs

if __name__ == '__main__':
    # --- Create a sample data dictionary for testing ---
    sample_data = {
        'Map Data': [{'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1963.56982422, 7806.85546875]), np.array([1963.52709961, 7843.18310547])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1966.96679688, 7806.84423828]), np.array([1966.98547363, 7843.69970703])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1966.93896484, 7785.59375]), np.array([1966.95495605, 7793.99853516]), np.array([1969.23913574, 7797.90966797]), np.array([1970.70812988, 7798.00439453])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1966.94824219, 7793.01367188]), np.array([1966.96679688, 7806.84423828])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1963.64428711, 7792.94677734]), np.array([1963.56982422, 7806.85546875])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1970.98742676, 7801.27636719]), np.array([1969.01513672, 7801.32470703]), np.array([1967.06616211, 7801.60839844]), np.array([1963.73632812, 7804.89355469]), np.array([1963.56982422, 7806.85546875])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1970.98742676, 7801.27636719]), np.array([1962.15856934, 7801.75683594])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1967.05712891, 7804.48193359]), np.array([1966.96679688, 7806.84423828])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1970.98742676, 7801.27636719]), np.array([1967.02539062, 7801.43017578]), np.array([1962.11303711, 7802.08984375])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1970.98742676, 7801.27636719]), np.array([1968.0501709, 7801.18896484]), np.array([1962.22180176, 7800.45507812])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1963.92077637, 7759.53564453]), np.array([1966.51672363, 7776.18798828]), np.array([1966.83496094, 7780.14111328]), np.array([1966.93896484, 7785.59375])]}, {'type': 'LaneCenter-SurfaceStreet', 'pos_xy': [np.array([1963.57446289, 7760.57958984]), np.array([1963.64428711, 7792.94677734])]}, {'type': 'RoadLine-BrokenSingleWhite', 'pos_xy': [np.array([1961.87744141, 7761.74658203]), np.array([1961.93127441, 7802.21337891])]}, {'type': 'RoadLine-SolidSingleWhite', 'pos_xy': [np.array([1965.40332031, 7785.50488281]), np.array([1965.16125488, 7843.51123047])]}, {'type': 'RoadEdgeBoundary', 'pos_xy': [np.array([1968.43725586, 7822.7265625]), np.array([1968.78320312, 7845.12158203])]}, {'type': 'RoadEdgeBoundary', 'pos_xy': [np.array([1968.4642334, 7806.51367188]), np.array([1968.43725586, 7822.7265625])]}, {'type': 'RoadEdgeBoundary', 'pos_xy': [np.array([1967.46484375, 7760.70117188]), np.array([1968.63195801, 7782.35400391]), np.array([1968.51171875, 7794.00048828]), np.array([1971.43725586, 7794.03027344])]}, {'type': 'Crosswalk', 'pos_xy': [np.array([1965.60144043, 7845.03515625]), np.array([1968.67199707, 7845.16992188])]}],
        'Ego Trajectory': {'type': 'Vehicle', 'trajectory': np.array([[-1.00000000e+00, -1.00000000e+00], [1.96526404e+03, 7.76571191e+03], [1.96557971e+03, 7.77292139e+03], [1.96608276e+03, 7.78193604e+03], [1.96643823e+03, 7.78937842e+03], [1.96669189e+03, 7.79597900e+03], [1.96683398e+03, 7.80195312e+03], [1.96690027e+03, 7.80714648e+03], [1.96693176e+03, 7.81174316e+03], [1.96695154e+03, 7.81586426e+03], [1.96697522e+03, 7.81946924e+03], [1.96698767e+03, 7.82249658e+03], [1.96699011e+03, 7.82487500e+03], [1.96699292e+03, 7.82656885e+03], [1.96699207e+03, 7.82779785e+03], [1.96698657e+03, 7.82872119e+03], [1.96698218e+03, 7.82952588e+03], [1.96697607e+03, 7.83041113e+03], [1.96697656e+03, 7.83131445e+03]])},
        'Nearby Agent Trajectories': {650: {'type': 'Vehicle', 'trajectory': np.array([[1966.87072754, 7828.43798828], [1966.87475586, 7828.45458984], [1966.86901855, 7828.484375], [1966.86706543, 7828.66162109], [1966.86401367, 7829.0703125], [1966.85656738, 7829.80566406], [1966.84875488, 7830.76269531], [1966.83544922, 7831.95849609], [1966.8236084, 7833.11181641], [1966.81616211, 7834.24609375], [1966.81835938, 7835.26904297], [1966.81335449, 7836.15917969], [1966.80432129, 7836.77832031], [1966.7989502, 7837.21044922], [1966.81237793, 7837.69873047], [1966.82092285, 7838.26416016], [1966.83056641, 7838.88818359], [1966.83239746, 7839.64648438], [1966.84216309, 7840.64794922]])}, 672: {'type': 'Vehicle', 'trajectory': np.array([[1964.07141113, 7790.80615234], [1964.06347656, 7795.94921875], [1964.04870605, 7800.78564453], [1964.0435791, 7805.06054688], [1964.01977539, 7809.04199219], [1964.00061035, 7812.54638672], [1963.9921875, 7815.69042969], [1963.96240234, 7818.52832031], [1963.95715332, 7821.07714844], [1963.95263672, 7823.296875], [1963.94311523, 7825.2109375], [1963.94128418, 7826.83349609], [1963.94030762, 7828.14355469], [1963.94372559, 7829.20605469], [1963.94592285, 7830.00097656], [1963.94299316, 7830.61865234], [1963.94763184, 7831.14111328], [1963.94812012, 7831.54931641], [1963.9543457, 7831.82324219]])}}
    }

    from convert_data_utils import convert_to_descriptor_format

    converted_data = convert_to_descriptor_format(sample_data)

    converted_data = converted_data[2]  # get the third time step for demonstration, which is the 10th second of original waymo motion

    # --- Generate and print the QA pairs ---
    generated_qa = generate_qa_for_timestep(converted_data)
    
    print("--- Generated QA Pairs ---")
    for i, qa in enumerate(generated_qa):
        print(f"\n{i+1}. Question: {qa['question']}")
        print(f"   Answer: {qa['answer']}")