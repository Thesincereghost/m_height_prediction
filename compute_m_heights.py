import sys
import pickle
import gzip
import os
import numpy as np
from itertools import combinations, product
from ortools.linear_solver import pywraplp
import time
import math
from concurrent.futures import ProcessPoolExecutor

def solve_m_height_lp_google(G, a, b, X, psi, debug=False):
    """
    Solves the m-height LP problem for a given tuple (a, b, X, psi) using Google's OR-Tools.
    """
    k, n = G.shape
    Y = list(set(range(n)) - set(X) - {a, b})  # Y = [n] \ X \ {a, b}
    tau = [a] + sorted(X) + [b] + sorted(Y)
    tau_inv = {tau[j]: j for j in range(n)}  # Inverse of τ

    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        raise Exception("Solver not created. Ensure OR-Tools is installed correctly.")

    u = [solver.NumVar(-solver.infinity(), solver.infinity(), f'u_{i}') for i in range(k)]

    objective = solver.Objective()
    for i in range(k):
        coefficient = float(psi[0] * G[i, a])
        objective.SetCoefficient(u[i], coefficient)
    objective.SetMaximization()

    for j in X:
        j_tau = tau_inv[j]
        constraint1 = solver.Constraint(-solver.infinity(), 0)
        constraint2 = solver.Constraint(-solver.infinity(), -1)
        for i in range(k):
            constraint1.SetCoefficient(u[i], float(psi[j_tau] * G[i, j] - psi[0] * G[i, a]))
            constraint2.SetCoefficient(u[i], float(-psi[j_tau] * G[i, j]))

    eq_constraint = solver.Constraint(1, 1)
    for i in range(k):
        eq_constraint.SetCoefficient(u[i], float(G[i, b]))

    for j in Y:
        constraint1 = solver.Constraint(-solver.infinity(), 1)
        constraint2 = solver.Constraint(-solver.infinity(), 1)
        for i in range(k):
            constraint1.SetCoefficient(u[i], float(G[i, j]))
            constraint2.SetCoefficient(u[i], float(-G[i, j]))

    status = solver.Solve()
    if status == pywraplp.Solver.INFEASIBLE:
        return 0
    elif status == pywraplp.Solver.UNBOUNDED:
        return float('inf')
    else:
        return objective.Value()

def compute_m_heights(G, lp_solve_function, max_m_value, debug=False):
    """
    Computes all m-heights of the code defined by the generator matrix G.
    """
    n = G.shape[1]
    m_heights = []

    for m in range(1, max_m_value + 1):
        max_hm = float('-inf')
        for a, b in product(range(n), repeat=2):
            if a != b:
                for X in combinations(set(range(n)) - {a, b}, m - 1):
                    for psi in product([-1, 1], repeat=m):
                        result = lp_solve_function(G, a, b, list(X), list(psi), debug=debug)
                        max_hm = max(max_hm, result)
        m_heights.append(max_hm)
        if max_hm == float('inf'):
            break

    # Fill remaining m-heights with infinity
    for m in range(len(m_heights), min(max_m_value + 1, n)):
        m_heights.append(float('inf'))

    return m_heights

def process_batch(batch_data):
    """
    Processes a batch of generator matrices, computes m-heights, and returns the results.
    """
    results = []
    for row in batch_data:
        G = np.array(row["generator_matrix"], dtype=np.float32)
        max_m_value = row["max_m_value"]

        k = G.shape[0]
        P = G[:, k:]  # Remove the kxk identity matrix from the front of G

        m_heights = compute_m_heights(G, solve_m_height_lp_google, max_m_value)

        results.append({
            "n": G.shape[1],
            "k": G.shape[0],
            "P_matrix": P.tolist(),
            "m_heights": m_heights
        })
    return results

def process_dataset(input_file, batch_size, num_workers):
    """
    Processes a generator matrix dataset in batches using multiprocessing,
    computes m-heights for each row, and saves the results to separate files.
    """
    try:
        with gzip.open(input_file, 'rb') as f:
            dataset = pickle.load(f)

        if not dataset:
            print("The dataset is empty.")
            return

        # os.makedirs("samples", exist_ok=True)

        # Create a subfolder under "samples" for the input file
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_folder = os.path.join("samples", base_name)
        os.makedirs(output_folder, exist_ok=True)

        total_samples = len(dataset)
        running_total = 0

        # Split the dataset into batches
        batches = [dataset[i:i + batch_size] for i in range(0, total_samples, batch_size)]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for batch_number, batch_data in enumerate(batches, start=1):
                # Process the batch
                results = executor.submit(process_batch, batch_data).result()

                # Save the batch results to a file
                # output_file = f"samples/{dataset[0]['n']}_{dataset[0]['k']}_batch_{batch_number}.pkl.gz"
                output_file = os.path.join(output_folder, f"batch_{batch_number}.pkl.gz")
                with gzip.open(output_file, 'wb') as f:
                    pickle.dump(results, f)

                running_total += len(batch_data)
                print(f"Processed batch {batch_number}, running total: {running_total}/{total_samples}")

        print("All batches have been processed and saved.")

    except FileNotFoundError:
        print(f"Error: File not found: {input_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python compute_m_heights.py <input_dataset_file> <batch_size> <num_workers>")
        sys.exit(1)

    input_file = sys.argv[1]
    batch_size = int(sys.argv[2])
    num_workers = int(sys.argv[3])

    process_dataset(input_file, batch_size, num_workers)