import os
import gzip
import pickle
from pathlib import Path

def combine_batches(samples_dirs, output_dir):
    """
    Combines batches from multiple samples directories into a single output directory.

    Args:
        samples_dirs (list): List of directories containing the samples.
        output_dir (str): Directory where the combined datasets will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect all unique dataset subfolder names
    dataset_names = set()
    for samples_dir in samples_dirs:
        for subfolder in os.listdir(samples_dir):
            dataset_names.add(subfolder)
    # dataset_names = set(['G_9_6_maxM3.pkl'])
    print(f"Found dataset names: {dataset_names}")
    # Process each dataset
    for dataset_name in dataset_names:
        combined_batches = []
        for samples_dir in samples_dirs:
            dataset_path = os.path.join(samples_dir, dataset_name)
            if os.path.exists(dataset_path) and os.path.isdir(dataset_path):
                for batch_file in sorted(os.listdir(dataset_path)):
                    batch_path = os.path.join(dataset_path, batch_file)
                    if batch_file.endswith(".pkl.gz"):
                        with gzip.open(batch_path, 'rb') as f:
                            batch_data = pickle.load(f)
                            combined_batches.extend(batch_data)

        # Save the combined dataset
        # output_dataset_path = os.path.join(output_dir, dataset_name)
        # os.makedirs(output_dataset_path, exist_ok=True)
        output_file = os.path.join(output_dir, dataset_name + "gz")
        with gzip.open(output_file, 'wb') as f:
            pickle.dump(combined_batches, f)
            # Print the size of the combined dataset
            print(f"Size of combined dataset '{dataset_name}': {len(combined_batches)} samples saved to {output_file}")
        # print(f"Combined dataset '{dataset_name}' saved to {output_file}")

if __name__ == "__main__":
    # List of directories containing the samples
    samples_dirs = ["samples", "samples_1", "samples_2" ]
    # Output directory for combined datasets
    output_dir = "samples_combined_1"

    combine_batches(samples_dirs, output_dir)