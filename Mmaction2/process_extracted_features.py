import os
import numpy as np
import torch # Import torch
from scipy.interpolate import interp1d
import mmengine # For loading/dumping .pkl files
from tqdm import tqdm # For progress bar

# --- CONFIGURATION ---

# 1. Define Input/Output Paths
#    Directory where individual vid_xxx.pkl files (containing Tensors) are saved
feature_dir = 'rgb_tarin_feat.pkl'
#    Your list of video IDs (train OR val)
video_list_file = 'anet_train_video.txt'
#    Output file for BMN
output_pkl_file = 'bmn_rgb_train_features_100.pkl'

# 2. Define Target Temporal Length for BMN
TARGET_TEMP_LEN = 100

# --- SCRIPT LOGIC ---

def load_video_list(filename):
    """Loads video IDs from a file (one ID per line, potentially with extra info)."""
    try:
        with open(filename, 'r') as f:
            # Read lines and take the first part (video ID) before any space
            video_ids = [line.strip().split(' ')[0] for line in f if line.strip()]
        print(f"Loaded {len(video_ids)} video IDs from {filename}")
        return video_ids
    except FileNotFoundError:
        print(f"ERROR: Video list file not found: {filename}")
        return None

def process_features(video_ids, input_dir, output_file, target_len):
    """Loads individual feature tensors, interpolates, and saves combined dict."""
    if video_ids is None:
        return

    processed_features = {}
    feature_dim = None # To store the dimensionality D

    print(f"Target temporal length: {target_len}")

    for video_id in tqdm(video_ids, desc="Processing Videos"):
        feature_path = os.path.join(input_dir, f"{video_id}.pkl")

        try:
            # Load the individual feature file - Should be a Tensor directly
            raw_features_tensor = mmengine.load(feature_path)

            # --- Convert to NumPy array (move to CPU if needed) ---
            if isinstance(raw_features_tensor, torch.Tensor):
                 if raw_features_tensor.is_cuda:
                     raw_features_tensor = raw_features_tensor.cpu()
                 raw_features_np = raw_features_tensor.numpy()
            elif isinstance(raw_features_tensor, np.ndarray):
                 raw_features_np = raw_features_tensor # Already numpy
            else:
                 print(f"Warning: Loaded object for {video_id} is not a Tensor or NumPy array. Type: {type(raw_features_tensor)}. Skipping.")
                 continue

            # --- Ensure it's 2D (T_original, D) - Handle potential extra dimensions ---
            if raw_features_np.ndim > 2:
                 # Example: If shape is (1, T, D), squeeze the first dim
                 if raw_features_np.shape[0] == 1:
                      raw_features_np = raw_features_np.squeeze(0)
                 else: # Add more specific handling if needed based on observed shapes
                      print(f"Warning: Feature array for {video_id} has unexpected dimensions {raw_features_np.shape}. Trying to reshape or skip.")
                      # Attempt a reasonable reshape or skip
                      # raw_features_np = raw_features_np.reshape(-1, raw_features_np.shape[-1]) # Example reshape
                      continue # Safer to skip if unsure

            if raw_features_np.ndim != 2 or raw_features_np.shape[0] == 0:
                 print(f"Warning: Feature array for {video_id} is not 2D or is empty after processing. Shape: {raw_features_np.shape}. Skipping.")
                 continue

            T_original, D = raw_features_np.shape

            if feature_dim is None:
                feature_dim = D
                print(f"Detected feature dimension (D): {feature_dim}")
            elif feature_dim != D:
                print(f"Warning: Inconsistent feature dimension for {video_id} (Expected {feature_dim}, Got {D}). Skipping.")
                continue

            # --- Temporal Interpolation ---
            if T_original == 1:
                # If only one feature vector, repeat it
                normalized_features = np.tile(raw_features_np, (target_len, 1))
            else:
                # Linear interpolation
                x_original = np.linspace(0, 1, T_original)
                x_target = np.linspace(0, 1, target_len)
                interpolator = interp1d(x_original, raw_features_np, axis=0, kind='linear', bounds_error=False, fill_value='extrapolate')
                normalized_features = interpolator(x_target)

            # Store the processed features (ensure float32 for consistency)
            processed_features[video_id] = normalized_features.astype(np.float32)

        except FileNotFoundError:
            print(f"Warning: Feature file not found: {feature_path}. Skipping video.")
        except Exception as e:
            print(f"Error processing {video_id}: {e}. Skipping video.")

    if not processed_features:
         print("No features were processed successfully.")
         return

    print(f"\nProcessed {len(processed_features)} videos.")
    print(f"Saving combined features to: {output_file}")
    mmengine.dump(processed_features, output_file)
    print("Done.")

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting feature post-processing...")
    video_ids_to_process = load_video_list(video_list_file)
    # Call process_features without the FEATURE_KEY argument
    process_features(
        video_ids_to_process,
        feature_dir,
        output_pkl_file,
        TARGET_TEMP_LEN
    )