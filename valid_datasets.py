import os
import gzip
import pickle

def read_datasets(folder="generator_matrices"):
    """
    Reads all datasets from the specified folder and prints the number of samples in each.
    """
    if not os.path.exists(folder):
        print(f"Folder '{folder}' does not exist.")
        return

    for file_name in os.listdir(folder):
        if file_name.endswith(".pkl.gz"):
            file_path = os.path.join(folder, file_name)
            with gzip.open(file_path, 'rb') as f:
                dataset = pickle.load(f)
                print(f"{file_name}: {len(dataset)} samples")

if __name__ == "__main__":
    read_datasets()
