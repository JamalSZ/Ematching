import numpy as np
import time
import tracemalloc
import multiprocessing as mp
import csv


# --- Algorithm Imports ---
from algs.iesj import *
from algs.ifi import *
from algs.bf import *
from algs.rnlj import *
from algs.fit import *


# --- Algorithm Wrappers ---
def algorithm_bf(T, e, n=0):
    _ = BF(T, e).lzcnl_numeric()

def algorithm_rnlj(T, e, n=0):
    _ = RNLJ(T, e).nljoin()

def algorithm_ifi(T, e, n=0):
    _ = IFI(T, e).LZ2_ifi()

def algorithm_iesj(series, e, n=0):
    _ = IESJ(series, e).iejoin()
    

def algorithm_fit(series, e, n=0):
    _ = FIT(series, e).run_fit()

def run_edge_case_experiment(params):
    alg, alg_name, N, series, e, run_id = params
    n = len(series)
    tracemalloc.start()
    start_time = time.perf_counter()
    _ = alg(series, e, n=n)
    end_time = time.perf_counter()
    current, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rt = end_time - start_time
    mem = peak_mem / (1024 * 1024)  # Convert to MB

    # Write results to CSV file for edge cases
    filename = f"Exp_results/Edge_Cases/{alg_name}_{N}.csv"
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([alg_name, len(series), e, rt, mem, run_id])

    return [alg_name, len(series), e, rt, mem, run_id]

def generate_edge_case_series(seed,n, N, case_type="random"):
    np.random.seed(seed)
    if case_type == "repetitive":
        return [N//2] * n  # All elements are the same
    elif case_type == "noisy":
        return np.random.randint(0, N+1, size=n).tolist()  # Random noisy series
    else:
        return np.random.randint(0, N+1, size=n).tolist()  # Regular random series

if __name__ == "__main__":
    algorithms = [algorithm_bf, algorithm_rnlj, algorithm_ifi, algorithm_iesj, algorithm_fit]
    alg_names = ["BF", "RNLJ", "IFI", "IESJ", "FIT"]
    n_values = [8000]  # Fixed n for edge case testing
    N_values = [32]  # Domain sizes for edge cases
    e_values_dict = {
        32: [0.01, 12]    
    }
    K = 1  # Number of times to repeat each experiment
    seed = 42
    

    with mp.Pool() as pool:
        for N in N_values:
            e_values = e_values_dict[N]
            data = generate_edge_case_series(seed,n_values[0], N)
            for e in e_values:    
                params_list = [
                    (alg, alg_name, N, data, e, run_id)
                    for alg, alg_name in zip(algorithms, alg_names)
                    for run_id in range(K)
                ]
                
                for _ in pool.imap_unordered(run_edge_case_experiment, params_list):
                    pass