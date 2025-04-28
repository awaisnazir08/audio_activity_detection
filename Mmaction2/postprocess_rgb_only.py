# Copyright (c) OpenMMLab. All rights reserved.
# Modified for RGB-only features without temporal pooling.
import argparse
import multiprocessing
import os
import os.path as osp
import warnings

import numpy as np
import torch  # Added for loading PyTorch Tensors
from mmengine import dump, load

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Feature Postprocessing (RGB Only)')
    parser.add_argument(
        '--feature_dir',
        required=True,
        help='Root directory containing the extracted .pkl RGB feature files')
    parser.add_argument(
        '--dest_dir',
        required=True,
        help='Destination directory to save the processed .csv files (e.g., mmaction_feat)')
    parser.add_argument(
        '--feature_dim',
        type=int,
        required=True,
        help='The dimensionality of your extracted RGB features (e.g., 2048 for TSN ResNet50)')
    parser.add_argument(
        '--output-format',
        default='csv',
        choices=['csv', 'pkl'],
        help='Output format for the processed features')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=16, # Adjust based on your CPU cores
        help='Number of processes to use for parallel processing')
    args = parser.parse_args()
    return args


def process_single_video(input_pkl_path, output_dir, output_format, expected_feature_dim):
    """
    Loads features from a .pkl file, ensures correct format, and saves
    in the desired output format without pooling.
    """
    try:
        # Load features (could be NumPy array or PyTorch Tensor)
        features = load(input_pkl_path)

        # --- Convert to NumPy and Validate Shape ---
        if isinstance(features, torch.Tensor):
            # Move to CPU if on GPU and convert to NumPy
            features = features.cpu().numpy()
        elif not isinstance(features, np.ndarray):
            warnings.warn(f"Skipping {input_pkl_path}: Loaded data is not a Tensor or NumPy array (type: {type(features)})")
            return

        # Handle potential extra dimensions (e.g., (N, 1, D) -> (N, D))
        if features.ndim == 3 and features.shape[1] == 1:
            features = features.squeeze(axis=1)
            # print(f"Note: Squeezed features for {input_pkl_path} from 3D to 2D")
        elif features.ndim != 2:
             warnings.warn(f"Skipping {input_pkl_path}: Expected 2D features (N, D) or squeezable 3D features (N, 1, D), but got shape {features.shape}")
             return

        # Validate feature dimension
        actual_feature_dim = features.shape[1]
        if actual_feature_dim != expected_feature_dim:
             warnings.warn(f"Skipping {input_pkl_path}: Expected feature dimension {expected_feature_dim} but got {actual_feature_dim}")
             return
        # --- End Validation ---


        # Determine output filename
        base_filename = osp.basename(input_pkl_path)
        if output_format == 'csv':
            output_filename = base_filename.replace('.pkl', '.csv')
        elif output_format == 'pkl':
             output_filename = base_filename # Keep .pkl extension
        else:
            # Should not happen due to argparse choices, but good practice
            raise ValueError(f"Unsupported output format: {output_format}")

        output_path = osp.join(output_dir, output_filename)

        # --- Save the processed features ---
        if output_format == 'pkl':
            dump(features, output_path)
        elif output_format == 'csv':
            # Create header line: f0,f1,...,f<D-1>
            header = ','.join([f'f{i}' for i in range(expected_feature_dim)])
            lines = [header]
            # Format each row (feature vector)
            for feature_vector in features:
                lines.append(','.join([f'{x:.6f}' for x in feature_vector])) # Use .6f for precision

            # Write to CSV file
            with open(output_path, 'w') as f:
                f.write('\n'.join(lines))

    except Exception as e:
        warnings.warn(f"Error processing {input_pkl_path}: {e}")


def main():
    args = parse_args()

    # Create destination directory if it doesn't exist
    os.makedirs(args.dest_dir, exist_ok=True)
    print(f"Ensured destination directory exists: {args.dest_dir}")

    # Find all .pkl files in the source directory
    feature_files = [
        f for f in os.listdir(args.feature_dir)
        if f.endswith('.pkl') and osp.isfile(osp.join(args.feature_dir, f))
    ]

    if not feature_files:
        print(f"Error: No .pkl files found in {args.feature_dir}")
        return

    print(f"Found {len(feature_files)} .pkl feature files to process.")

    # Prepare arguments for parallel processing
    tasks = [
        (osp.join(args.feature_dir, fname), args.dest_dir, args.output_format, args.feature_dim)
        for fname in feature_files
    ]

    # Use multiprocessing pool
    print(f"Starting processing with {args.num_workers} workers...")
    pool = multiprocessing.Pool(args.num_workers)
    # Use starmap to pass multiple arguments to the worker function
    pool.starmap(process_single_video, tasks)
    pool.close()
    pool.join()

    print(f"Processing complete. Output files should be in {args.dest_dir}")
    print("Please check for any warnings printed above about skipped files or errors.")


if __name__ == '__main__':
    main()