import os
import pickle
import gzip
import numpy as np
from generate_datasets import generate_random_P, construct_generator_matrix, generate_dataset

def load_existing_matrices(folder_paths, n, k, max_m_value):
    """
    Loads the existing matrix for a specific (n, k, max_m_value) from the specified folders.
    """
    existing_Ps = set()
    file_name = f"G_{n}_{k}_maxM{max_m_value}.pkl.gz"

    for folder_path in folder_paths:
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            with gzip.open(file_path, 'rb') as f:
                dataset = pickle.load(f)
                for entry in dataset:
                    G = np.array(entry["generator_matrix"])
                    P = G[:, G.shape[1] - G.shape[0]:]  # Extract the P matrix
                    P_columns = frozenset(tuple(col) for col in P.T)
                    existing_Ps.add(P_columns)
            print(f"Loaded matrices from {file_path}.")
            break  # Stop searching once the file is found
    return existing_Ps

def generate_unique_datasets(n, k, num_samples, max_m_value, output_folder, existing_folders):
    """
    Generates datasets ensuring no duplicates with previously generated datasets.
    """
    os.makedirs(output_folder, exist_ok=True)
    existing_Ps = load_existing_matrices(existing_folders, n, k, max_m_value)
    print(f"Generating unique datasets for n={n}, k={k}, max_m_value={max_m_value}...")

    dataset = []
    for P in generate_random_P(n, k, existing_Ps, num_samples):
        G = construct_generator_matrix(n, k, P)
        dataset.append({
            "generator_matrix": G.tolist(),
            "max_m_value": max_m_value
        })

    file_name = f"{output_folder}/G_{n}_{k}_maxM{max_m_value}.pkl.gz"
    with gzip.open(file_name, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"Unique dataset saved to {file_name}")
    return file_name

if __name__ == "__main__":
    existing_folders = ["generator_matrices"]
    output_folder = "generator_matrices_1"
    combinations = [
        (9, 4, 5, 25000),
        (9, 5, 4, 50000),
        (9, 6, 3, 50000),
        (10, 4, 6, 25000),
        (10, 5, 5, 25000),
        (10, 6, 6, 25000)
    ]

    for n, k, max_m_value, num_samples in combinations:
        generate_unique_datasets(n, k, num_samples, max_m_value, output_folder, existing_folders)
