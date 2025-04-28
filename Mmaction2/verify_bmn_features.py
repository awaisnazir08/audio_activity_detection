import os
import numpy as np
import mmengine # For loading .pkl files
from tqdm import tqdm # Optional: for progress bar on large datasets

# --- CONFIGURATION ---

# 1. Path to the final BMN feature file you want to verify
feature_pkl_file = 'bmn_rgb_train_features_100.pkl' # ADJUST FOR TRAIN/VAL

# 2. Path to the corresponding video list file used to generate this feature file
#    (This helps verify all expected videos are present)
video_list_file = 'anet_train_video.txt' # ADJUST FOR TRAIN/VAL

# 3. Expected temporal dimension (after interpolation)
EXPECTED_T = 100

# 4. Expected feature dimension (e.g., 2048 for ResNet50 backbone)
EXPECTED_D = 2048

# 5. Expected data type
EXPECTED_DTYPE = np.float32

# --- VERIFICATION LOGIC ---

def load_video_list(filename):
    """Loads video IDs from a file (one ID per line, first part before space)."""
    try:
        with open(filename, 'r') as f:
            video_ids = set(line.strip().split(' ')[0] for line in f if line.strip())
        print(f"Loaded {len(video_ids)} unique video IDs from {filename}")
        return video_ids
    except FileNotFoundError:
        print(f"ERROR: Video list file not found: {filename}")
        return None

def verify_features(pkl_file, list_file, expected_t, expected_d, expected_dtype):
    """Loads the final feature dict and checks its contents."""
    print(f"\n--- Verifying File: {pkl_file} ---")

    # 1. Check if file exists
    if not os.path.exists(pkl_file):
        print(f"ERROR: Feature file not found: {pkl_file}")
        return False

    # 2. Load the file
    try:
        data = mmengine.load(pkl_file)
        print("Successfully loaded feature file.")
    except Exception as e:
        print(f"ERROR: Failed to load feature file: {e}")
        return False

    # 3. Check if it's a dictionary
    if not isinstance(data, dict):
        print(f"ERROR: Loaded data is not a dictionary (Type: {type(data)}). Expected dict.")
        return False
    print(f"Data structure is a dictionary with {len(data)} entries.")

    # 4. Load corresponding video list
    expected_video_ids = load_video_list(list_file)
    if expected_video_ids is None:
        return False # Error already printed

    # 5. Compare keys in dict with video IDs in list
    feature_keys = set(data.keys())
    if feature_keys != expected_video_ids:
        print("ERROR: Mismatch between video IDs in the list file and keys in the feature file!")
        missing_in_features = expected_video_ids - feature_keys
        extra_in_features = feature_keys - expected_video_ids
        if missing_in_features:
            print(f"  - IDs in list but MISSING in feature file: {sorted(list(missing_in_features))}")
        if extra_in_features:
            print(f"  - Keys in feature file but NOT in list: {sorted(list(extra_in_features))}")
        # Decide if this is fatal - often it is.
        # return False # Uncomment if mismatch should halt verification

    # 6. Check individual features (shape, dtype, content)
    print(f"Checking individual features (expecting shape=({expected_t}, {expected_d}), dtype={expected_dtype})...")
    all_checks_passed = True
    for video_id, feature_array in tqdm(data.items(), desc="Verifying entries"):
        # Check type
        if not isinstance(feature_array, np.ndarray):
            print(f"  - ERROR [{video_id}]: Feature is not a NumPy array (Type: {type(feature_array)})")
            all_checks_passed = False
            continue # Skip further checks for this entry

        # Check shape
        if feature_array.shape != (expected_t, expected_d):
            print(f"  - ERROR [{video_id}]: Incorrect shape. Expected ({expected_t}, {expected_d}), Got {feature_array.shape}")
            all_checks_passed = False

        # Check dtype
        if feature_array.dtype != expected_dtype:
            print(f"  - ERROR [{video_id}]: Incorrect dtype. Expected {expected_dtype}, Got {feature_array.dtype}")
            all_checks_passed = False

        # Check for NaN/Inf (optional but good practice)
        if np.isnan(feature_array).any():
            print(f"  - WARNING [{video_id}]: Feature array contains NaN values.")
            # Decide if this is acceptable or an error
            # all_checks_passed = False
        if np.isinf(feature_array).any():
            print(f"  - WARNING [{video_id}]: Feature array contains Inf values.")
            # Decide if this is acceptable or an error
            # all_checks_passed = False

    # 7. Final Summary
    print("-" * (len(f"--- Verifying File: {pkl_file} ---"))) # Match title length
    if all_checks_passed:
        print("SUCCESS: All checks passed!")
        return True
    else:
        print("FAILURE: One or more checks failed. Please review the errors above.")
        return False

# --- Main Execution ---
if __name__ == "__main__":
    verify_features(
        feature_pkl_file,
        video_list_file,
        EXPECTED_T,
        EXPECTED_D,
        EXPECTED_DTYPE
    )
    # Example: How to verify the validation set afterwards
    # print("\n\nNow verifying validation set...")
    # verify_features(
    #     '../../../data/ActivityNet/bmn_rgb_val_features_100.pkl', # CHANGE PATH
    #     '../../../data/ActivityNet/anet_val_video.txt',          # CHANGE PATH
    #     EXPECTED_T,
    #     EXPECTED_D,
    #     EXPECTED_DTYPE
    # )