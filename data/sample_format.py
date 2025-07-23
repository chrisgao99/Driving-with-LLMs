import pickle
import os

def print_sample(sample, index):
    """Helper function to print a single sample in a readable format."""
    print(f"\n=== Sample {index} ===")
    if isinstance(sample, dict):
        for key, value in sample.items():
            print(f"{key}: {value}")
    else:
        print(f"Sample content: {sample}")

def main():
    # Path to the pickle file
    pkl_path = "vqa_test_1k.pkl"
    
    # Check if file exists
    if not os.path.exists(pkl_path):
        print(f"Error: File {pkl_path} not found.")
        return
    
    try:
        # Load the pickle file
        with open(pkl_path, "rb") as f:
            dataset = pickle.load(f)
        
        # Check dataset type and length
        print(f"Dataset type: {type(dataset)}")
        if hasattr(dataset, "__len__"):
            print(f"Number of samples: {len(dataset)}")
        
        # Iterate through samples
        if isinstance(dataset, (list, tuple)):
            for i, sample in enumerate(dataset):
                print_sample(sample, i)
        elif isinstance(dataset, dict):
            for i, (key, sample) in enumerate(dataset.items()):
                print_sample({key: sample}, i)
        elif hasattr(dataset, "__getitem__") and hasattr(dataset, "__len__"):
            for i in range(len(dataset)):
                sample = dataset[i]
                print_sample(sample, i)
        else:
            print("Error: Unknown dataset format. Cannot iterate through samples.")
        
    except Exception as e:
        print(f"Error loading or processing {pkl_path}: {str(e)}")

if __name__ == "__main__":
    main()