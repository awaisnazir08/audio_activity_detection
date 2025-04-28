import json
import os
import numpy as np
import torch  # Import PyTorch
# Use pickle if mmengine is not readily available in your environment
# import pickle
from mmengine import load  # Preferred way in mm* ecosystem

# --- Configuration ---
# Path to your annotation file (the one WITHOUT feature_frame yet)
annotation_file_path = 'anet_train.json' # Using your filename
# Path to the directory where you saved the .pkl feature files
features_dir = 'rgb_tsn_training_features' # Using your directory name
# Path where you want to save the UPDATED annotation file
output_annotation_path = 'anet_train_updated.json' # New output filename
# --- End Configuration ---

print(f"Loading original annotations from: {annotation_file_path}")
with open(annotation_file_path, 'r') as f:
    annotation_data = json.load(f)

print(f"Processing {len(annotation_data)} videos...")
missing_features = []
updated_count = 0

# Iterate through each video ID in the annotation file
# Use list(annotation_data.keys()) to avoid issues if modifying dict during iteration (though we aren't here)
for video_id in list(annotation_data.keys()):
    # Assuming video IDs in JSON might have 'v_' prefix but pkl files don't, or vice-versa
    # Adjust filename construction if needed based on your exact naming
    # Example: If JSON has "v_vid_707" and pkl has "vid_707.pkl"
    # pkl_video_id = video_id.replace('v_', '') # Adjust prefix if needed
    pkl_video_id = video_id # Assuming JSON key matches pkl filename base
    pkl_filename = f"{pkl_video_id}.pkl"
    pkl_path = os.path.join(features_dir, pkl_filename)

    try:
        # Load the feature file
        features = load(pkl_path)

        feature_frame_count = -1 # Initialize

        # Check if it's a numpy array OR a torch Tensor
        if isinstance(features, np.ndarray):
            feature_frame_count = features.shape[0]
        elif isinstance(features, torch.Tensor): # <-- ADDED CHECK FOR TENSOR
            feature_frame_count = features.shape[0] # Get shape[0] same way
        else:
            print(f"Warning: Feature data for {video_id} is not a NumPy array or Torch Tensor (type: {type(features)}). Skipping.")
            missing_features.append(video_id + " (unexpected format)")
            continue # Skip to next video if format is wrong

        # Add/update the feature_frame key in the dictionary
        if video_id in annotation_data: # Ensure key exists before updating
             annotation_data[video_id]['feature_frame'] = feature_frame_count
             updated_count += 1
        else:
             print(f"Warning: Video ID {video_id} found in features but not in annotation JSON. Skipping update.")
             missing_features.append(video_id + " (not in JSON)")


        # Optional: print progress
        # if updated_count % 100 == 0:
        #     print(f"Processed {updated_count} videos...")


    except FileNotFoundError:
        print(f"Warning: Feature file not found for video {video_id} (expected at {pkl_path}). Skipping.")
        missing_features.append(video_id + " (file not found)")
    except Exception as e:
        print(f"Error processing {video_id} at {pkl_path}: {e}")
        missing_features.append(video_id + f" (error: {e})")


print("-" * 20)
print(f"Finished processing.")
print(f"Successfully updated feature_frame for {updated_count} videos.")

if missing_features:
    print("\nCould not process or encountered issues with the following videos:")
    for missing in missing_features:
        print(f"- {missing}")

print(f"\nSaving updated annotations to: {output_annotation_path}")
# Save the modified dictionary to a new JSON file
with open(output_annotation_path, 'w') as f:
    json.dump(annotation_data, f, indent=4) # indent=4 makes it readable

print("Done.")