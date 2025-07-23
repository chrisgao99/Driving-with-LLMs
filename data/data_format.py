import pickle
import torch

print(torch.__version__)  # Check PyTorch version

# Path to the pickle file
file_path = "vqa_test_1k.pkl" #one data has 4 keys['frame_num', 'observation', 'input_prompt', 'response_content']
file_path = "vqa_train_10k.pkl"  # Adjust the path as needed

# Load the pickle file
with open(file_path, "rb") as f:
    data = pickle.load(f)

# Inspect the contents
print(type(data))  # Check the data type (e.g., list, dict, DataFrame)
print(len(data))  # Print the data (or a subset if it's large)

for key, value in data[3].items():
    if key == "observation":
        #Observation data:  dict_keys(['ego_vehicle_descriptor', 'liable_vehicles', 'pedestrian_descriptors', 'route_descriptors', 'vehicle_descriptors'])
        # ego_vehicle_descriptor torch.Size([31])
        # liable_vehicles torch.Size([30]) True or False
        # pedestrian_descriptors torch.Size([20, 9]) zero rows at the end
        # route_descriptors torch.Size([30, 17])
        # vehicle_descriptors torch.Size([30, 33]) zero rows at the end
        print("Observation data: ",value.keys())  # Print keys of the observation data
        for obs_key, obs_value in value.items():
            print(f"{obs_key}: {obs_value}")
    else:
        print(f"{key}: {value}")  # Print each key-value pair in the first item

# If it's a list or dict, you can explore further
if isinstance(data[0], list):
    print("Number of items:", len(data[0]))
    print("First item:", data[0][0])
elif isinstance(data[0], dict):
    print("Keys:", data[0].keys())
#     print("Sample value:", list(data[0].values())[0])