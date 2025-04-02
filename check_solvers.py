
import torch
import numpy as np
from itertools import combinations, product
from concurrent.futures import ThreadPoolExecutor
import math
from scipy.optimize import linprog
import time
from ortools.linear_solver import pywraplp

def solve_m_height_lp_scipy(G, a, b, X, psi, debug = False):
    """
    Solves the m-height LP problem for a given tuple (a, b, X, psi).
    
    Parameters:
        G (numpy.ndarray): The k x n generator matrix.
        a (int): Index a in [n].
        b (int): Index b in [n] \ {a}.
        X (list): Subset X ⊆ [n] \ {a, b} with |X| = m - 1.
        psi (list): Binary vector of length m with elements in {-1, 1}.
    
    Returns:
        float: The optimal objective value of the LP problem.
               Returns 0 if infeasible, ∞ if unbounded.
    """
    k, n = G.shape
    Y = list(set(range(n)) - set(X) - {a, b})  # Y = [n] \ X \ {a, b}
    
    # Define the permutation τ
    tau = [a] + sorted(X) + [b] + sorted(Y)
    tau_inv = {tau[j]: j for j in range(n)}  # Inverse of τ
    
    # Objective function: maximize ∑(s0 * gi,a * ui)
    c = -psi[0] * G[:, a]  # Negative for maximization
    
    # Constraints
    A = []
    b_ub = []
    
    # Constraints for j ∈ X
    for j in X:
        j_tau = tau_inv[j]
        A.append(psi[j_tau] * G[:, j] - psi[0] * G[:, a])
        b_ub.append(0)
        A.append(-psi[j_tau] * G[:, j])
        b_ub.append(-1)
    
    # Equality constraint for j = b
    A_eq = [G[:, b]]
    b_eq = [1]
    
    # Constraints for j ∈ Y
    for j in Y:
        A.append(G[:, j])
        b_ub.append(1)
        A.append(-G[:, j])
        b_ub.append(1)
    
    # Solve the LP problem
    result = linprog(c, A_ub=A, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method="highs")
    
    if debug:
        print(f"n={n}, k={k}, a={a}, b={b}, X={X}, Y={Y}, tau={tau},\n tau_inv={tau_inv}")
        print(f"G={G},\n psi={psi},\n c={c},\n A={A},\n b_ub={b_ub},\n A_eq={A_eq},\n b_eq={b_eq}")
    if result.status == 2:  # Infeasible
        return 0
    elif result.status == 3:  # Unbounded
        return float('inf')
    else:
        return -result.fun  # Convert back to maximization value

def solve_m_height_lp_google(G, a, b, X, psi, debug=False):
    """
    Solves the m-height LP problem for a given tuple (a, b, X, psi) using Google's OR-Tools.
    
    Parameters:
        G (numpy.ndarray): The k x n generator matrix.
        a (int): Index a in [n].
        b (int): Index b in [n] \ {a}.
        X (list): Subset X ⊆ [n] \ {a, b} with |X| = m - 1.
        psi (list): Binary vector of length m with elements in {-1, 1}.
    
    Returns:
        float: The optimal objective value of the LP problem.
               Returns 0 if infeasible, ∞ if unbounded.
    """
    k, n = G.shape
    Y = list(set(range(n)) - set(X) - {a, b})  # Y = [n] \ X \ {a, b}
    
    # Define the permutation τ
    tau = [a] + sorted(X) + [b] + sorted(Y)
    tau_inv = {tau[j]: j for j in range(n)}  # Inverse of τ
    
    # Create the solver
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        raise Exception("Solver not created. Ensure OR-Tools is installed correctly.")
    
    # Variables
    u = [solver.NumVar(-solver.infinity(), solver.infinity(), f'u_{i}') for i in range(k)]
    
    # Objective function: maximize ∑(s0 * gi,a * ui)
    objective = solver.Objective()
    for i in range(k):
        coefficient = float(psi[0] * G[i, a])  # Ensure the coefficient is a Python float
        objective.SetCoefficient(u[i], coefficient)  # Negative for maximization
    objective.SetMaximization()
    
    # Constraints
    # Constraints for j ∈ X
    for j in X:
        j_tau = tau_inv[j]
        constraint1 = solver.Constraint(-solver.infinity(), 0)
        constraint2 = solver.Constraint(-solver.infinity(), -1)
        for i in range(k):
            constraint1.SetCoefficient(u[i], float(psi[j_tau] * G[i, j] - psi[0] * G[i, a]))
            constraint2.SetCoefficient(u[i], float(-psi[j_tau] * G[i, j]))
    
    # Equality constraint for j = b
    eq_constraint = solver.Constraint(1, 1)
    for i in range(k):
        eq_constraint.SetCoefficient(u[i], float(G[i, b]))
    
    # Constraints for j ∈ Y
    for j in Y:
        constraint1 = solver.Constraint(-solver.infinity(), 1)
        constraint2 = solver.Constraint(-solver.infinity(), 1)
        for i in range(k):
            constraint1.SetCoefficient(u[i], float(G[i, j]))
            constraint2.SetCoefficient(u[i], float(-G[i, j]))
    
    # Solve the LP problem
    status = solver.Solve()
    
    if debug:
        print(f"n={n}, k={k}, a={a}, b={b}, X={X}, Y={Y}, tau={tau},\n tau_inv={tau_inv}")
        print(f"G={G},\n psi={psi}")
    
    if status == pywraplp.Solver.INFEASIBLE:
        return 0
    elif status == pywraplp.Solver.UNBOUNDED:
        return float('inf')
    else:
        return objective.Value()  # Return the maximized value
    
def compute_m_heights(G, lp_solve_function, max_m_value, debug=False):
    """
    Computes all m-heights of the code defined by the generator matrix G.
    
    Parameters:
        G (numpy.ndarray): The k x n generator matrix.
        lp_solve_function (function): Function to solve the LP problem.
        max_m_value (int): Maximum allowed value for m.
        debug (bool): If True, prints debug information.

    Returns:
        dict: Dictionary of m-heights.
    """
    start_time = time.time()

    n = G.shape[1]
    m_heights = {0: 1}
    num_tuples_expected = 0
    num_tuples_processed = 0

    # Restrict m based on max_m_value
    for m in range(1, min(max_m_value+1, n)):
        max_hm = float('-inf')

        for a, b in product(range(n), repeat=2):
            if a != b:
                for X in combinations(set(range(n)) - {a, b}, m - 1):
                    for psi in product([-1, 1], repeat=m):
                        num_tuples_processed += 1
                        
                        # Call the LP solver
                        result = lp_solve_function(G, a, b, list(X), list(psi), debug=debug)
                        max_hm = max(max_hm, result)

        m_heights[m] = max_hm
        num_tuples_expected += n * (n - 1) * math.comb(n - 2, m - 1) * (2 ** m)

        if max_hm == float('inf'):
            break

    # Fill remaining m-heights with infinity
    for m in range(len(m_heights), n):
        m_heights[m] = float('inf')

    end_time = time.time()
    elapsed_time = end_time - start_time

    if debug:
        print(f"n = {n}, k = {G.shape[0]}")
        print(f"Expected number of tuples: {num_tuples_expected}")
        print(f"Processed {num_tuples_processed} tuples")
        print(f"Total computation time: {elapsed_time:.6f} seconds")

    return m_heights

if __name__ == "__main__":
    matrices ={'G^6_9,3': torch.tensor([
        [0.992, -0.026, -0.019, -0.08, -0.487, 0.645, 0.473, 0.598, -0.732],
        [0.002, 0.969, 0.032, 0.291, -0.671, 0.208, -0.837, -0.398, 1.118],
        [-0.006, -0.013, 1.014, -0.805, 1.011, 1.518, -0.114, -0.347, -0.408]
    ])}
    print("Computing m-heights using scipy solver...")
    start_time = time.time()
    for name, matrix in matrices.items():
        max_m_value = 6
        m_heights = compute_m_heights(matrix, lp_solve_function= solve_m_height_lp_google, max_m_value = max_m_value, debug=False)
        print(f"m-heights for {name}:")
        for m, height in m_heights.items():
            print(f"  {m}-height: {height}")
        print()
    end_time = time.time()
    print(f"Total computation time: {end_time - start_time:.6f} seconds")
    print("Computing m-heights using google solver...")
    start_time = time.time()
    for name, matrix in matrices.items():
        max_m_value = 6
        m_heights = compute_m_heights(matrix, lp_solve_function= solve_m_height_lp_scipy, max_m_value = max_m_value, debug=False)
        print(f"m-heights for {name}:")
        for m, height in m_heights.items():
            print(f"  {m}-height: {height}")
        print()
    end_time = time.time()
    print(f"Total computation time: {end_time - start_time:.6f} seconds")

