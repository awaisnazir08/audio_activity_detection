import mmengine
import numpy as np
import torch # Import torch

# --- CONFIGURATION ---
# CHANGE THIS to the path of ONE of the .pkl files
sample_feature_file = 'rgb_tarin_feat.pkl/vid_1001.pkl' # Use the one that worked

# --- Load and Inspect ---
try:
    # Load the file, it should directly be a tensor now
    feature_tensor = mmengine.load(sample_feature_file)

    print(f"Successfully loaded: {sample_feature_file}")

    # Check if it's a tensor and print info
    if isinstance(feature_tensor, torch.Tensor):
        print("Loaded object is a PyTorch Tensor.")
        print("Shape:", feature_tensor.shape)
        print("Dtype:", feature_tensor.dtype)
        print("Device:", feature_tensor.device) # Might be on GPU if not moved
    elif isinstance(feature_tensor, np.ndarray):
        print("Loaded object is a NumPy array.")
        print("Shape:", feature_tensor.shape)
        print("Dtype:", feature_tensor.dtype)
    else:
        print("Loaded object is of unexpected type:", type(feature_tensor))

except FileNotFoundError:
    print(f"ERROR: File not found - {sample_feature_file}")
    print("Please ensure the path is correct and the feature extraction script ran successfully.")
except Exception as e:
    print(f"An error occurred: {e}")