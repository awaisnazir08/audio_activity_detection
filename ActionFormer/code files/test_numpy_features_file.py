import numpy as np
import os # Import os to check if file exists before trying to load

# --- User Configuration ---
# IMPORTANT: Replace this with the actual path to ONE of the .npy files you created
# For example: npy_file_path = "./rally_features_i3d_stride4/my_rally_video_1.npy"
npy_file_path = "rally_features_i3d_stride_4/vid_1010.npy"

# --- Verification ---

print(f"Attempting to load: {npy_file_path}")

# Check if the file exists first
if not os.path.exists(npy_file_path):
    print(f"Error: File not found at '{npy_file_path}'")
    print("Please double-check the path and make sure the feature extraction script ran successfully.")
else:
    try:
        # Load the .npy file
        # allow_pickle=False is generally safer if you know it's a standard numpy array
        loaded_data = np.load(npy_file_path, allow_pickle=False)

        print("\n--- File Loaded Successfully ---")

        # 1. Verify Shape
        print(f"\nShape of the loaded array: {loaded_data.shape}")

        # Check number of dimensions and specific dimensions
        if loaded_data.ndim == 2:
            num_vectors = loaded_data.shape[0]
            feature_dim = loaded_data.shape[1]
            print(f" -> Temporal Length (T - number of feature vectors): {num_vectors}")
            print(f" -> Feature Dimension (C - size of each vector): {feature_dim}")

            # Explicitly check if the feature dimension is 2048
            if feature_dim == 2048:
                print(" -> Feature dimension (2048) is CORRECT for the ActionFormer config.")
            else:
                print(f" -> !!! WARNING !!!: Feature dimension is {feature_dim}, but ActionFormer config expects 2048!")
                print("    Fine-tuning might fail or perform poorly.")
        else:
            print(f" -> !!! WARNING !!!: Expected 2 dimensions (T, C), but found {loaded_data.ndim}.")
            print("    The data structure might be incorrect.")


        # 2. Verify Data Type
        print(f"\nData Type (dtype): {loaded_data.dtype}")
        # (Usually float32 or similar for features)


        # 3. Print a small sample (Optional, but helpful)
        # Check if the array is not empty and has 2 dimensions before sampling
        if loaded_data.ndim == 2 and loaded_data.shape[0] > 0:
            print(f"\nSample of the first feature vector (first 10 values):")
            # Print only the first 10 elements of the first vector for brevity
            print(loaded_data[0, :10])

            # Optionally print more samples if needed
            # if loaded_data.shape[0] > 2: # Check if there are at least 3 vectors
            #      print(f"\nSample of the first 3 feature vectors (first 10 values each):")
            #      print(loaded_data[:3, :10])

    except FileNotFoundError:
        # This is redundant due to the os.path.exists check, but good practice
        print(f"Error: File not found at '{npy_file_path}' during loading.")
    except Exception as e:
        print(f"\nError loading or processing file '{npy_file_path}': {e}")
        print("The file might be corrupted or not a valid NumPy file.")