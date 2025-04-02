import torch
import math
import pickle
import gzip
import os
from concurrent.futures import ProcessPoolExecutor

def generate_random_P(n, k, existing_Ps, num_samples):
    """
    Generates unique random k x (n-k) matrices P with real values bounded between -100 and 100.
    The number of unique matrices to generate is num_samples / (n-k)!.
    Matrices with the same columns but in a different order are considered duplicates.
    """
    target_count = num_samples // math.factorial(n - k)
    generated_count = 0

    while generated_count < target_count:
        P = torch.empty((k, n - k), dtype=torch.float32).uniform_(-100, 100)  # Real-valued matrix
        # Check for duplicates considering column permutations
        P_columns = frozenset(tuple(col.tolist()) for col in P.T)
        if P_columns not in existing_Ps:
            existing_Ps.add(P_columns)
            generated_count += 1
            yield P

def construct_generator_matrix(n, k, P):
    """
    Constructs the systematic k×n generator matrix G given the k×(n−k) matrix P.
    """
    if P.shape != (k, n - k):
        raise ValueError(f"P must have shape ({k}, {n-k}), but got {P.shape}")
    
    I_k = torch.eye(k, dtype=P.dtype)
    G = torch.cat((I_k, P), dim=1)
    return G

def generate_dataset(n, k, num_samples, max_m_value):
    """
    Generates a dataset of random P matrices and saves the generator matrices.
    Each row corresponds to one generator matrix and its associated max_m_value.
    Saves the dataset to a compressed file named G_n_k.
    """
    dataset = []
    existing_Ps = set()

    for P in generate_random_P(n, k, existing_Ps, num_samples):
        G = construct_generator_matrix(n, k, P)
        dataset.append({
            "generator_matrix": G.numpy().tolist(),
            "max_m_value": max_m_value
        })
    
    # Ensure the folder exists
    os.makedirs("generator_matrices", exist_ok=True)
    
    # Save the dataset to a compressed file
    file_name = f"generator_matrices/G_{n}_{k}_maxM{max_m_value}.pkl.gz"
    with gzip.open(file_name, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"Dataset saved to {file_name}")
    return file_name

def process_combination(args):
    """
    Helper function to process a single combination of (n, k, max_m_value).
    """
    n, k, max_m_value, num_samples = args
    print(f"Generating dataset for n={n}, k={k}, max_m_value={max_m_value}...")
    return generate_dataset(n, k, num_samples, max_m_value)

if __name__ == "__main__":
    combinations = [
        (9, 4, 5),
        (9, 5, 4),
        (9, 6, 3),
        (10, 4, 6),
        (10, 5, 5),
        (10, 6, 6)
    ]
    num_samples = 10000

    # Prepare arguments for multiprocessing
    args_list = [(n, k, max_m_value, num_samples) for n, k, max_m_value in combinations]

    # Use ProcessPoolExecutor with max_workers=4
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_combination, args_list))

    print("All datasets have been generated.")