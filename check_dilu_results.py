import json
import matplotlib.pyplot as plt
import numpy as np

result_path = "/p/liverobotics/Rui/Driving-with-LLMs/dilu_prediction.json"

with open(result_path, "r") as f:
    data = [json.loads(line) for line in f]

for item in data:
    sid = item['sid']
    ego_id = item['ego_id']
    predicted_trajectory = item['predicted_trajectory']
    ground_truth_trajectory = item['ground_truth_trajectory']

    gt_points = []
    for point in ground_truth_trajectory:
        if point[0] > 0 and point[1] > 0:
            gt_points.append(point)

    llm_points = [gt_points[0]]
    for point in predicted_trajectory[1:]:
        if point[0] > 0 and point[1] > 0:
            llm_points.append(point)

    gt_points = np.array(gt_points)
    llm_points = np.array(llm_points)
    min_len = min(len(gt_points), len(llm_points))
    gt_points = gt_points[:min_len]
    llm_points = llm_points[:min_len]
    # Plotting the trajectories
    plt.figure(figsize=(10, 6))
    plt.plot(gt_points[:, 0], gt_points[:, 1], marker='o', label='Ground Truth Trajectory', color='blue')
    plt.plot(llm_points[:, 0], llm_points[:, 1], marker='x', label='Predicted Trajectory', color='red')
    plt.legend()
    plt.savefig(f"trajectory_{sid}_{ego_id}.png")
    ade = np.mean(np.linalg.norm(gt_points - llm_points, axis=1))
    print(f"Average Displacement Error (ADE) for Scenario {sid}, Ego ID {ego_id}: {ade:.2f}m")
