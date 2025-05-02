import multiprocessing as mp
from tqdm import tqdm
import time
import tracemalloc
import csv
import numpy as np

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

# --- Experiment Execution ---
def run_experiment(params):
    alg, alg_name, N, series, e, run_id, lock = params
    n = len(series)

    tracemalloc.start()
    start_time = time.perf_counter()
    alg(series, e, n=n)
    end_time = time.perf_counter()
    current, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rt = end_time - start_time
    mem = peak_mem / (1024 * 1024)  # MB

    result = [alg_name, len(series), e, rt, mem, run_id]
    filename = f"Exp_results/Synthetic/{alg_name}_{N}.csv"

    # Ensure safe writing
    with lock:
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(result)
        print(f"[✓] {alg_name} finished | n={n} | N={N} | e={e} | run={run_id} | time={rt:.2f}s | mem={mem:.2f}MB")

    return result


# --- Time Series Generator ---
def generate_time_series(seed, n, N):
    np.random.seed(seed)
    return np.random.randint(0, N + 1, size=n).tolist()

# --- Main Execution ---
import multiprocessing as mp
import math

def main():
    algorithms = [
        (algorithm_bf, "BF"),
        (algorithm_rnlj, "RNLJ"),
        (algorithm_ifi, "IFI"),
        (algorithm_iesj, "IESJ"),
        (algorithm_fit, "FIT")
    ]

    n_values = [16000]
    N_values = [128]
    e_values_dict = {
        32: [0.01, 0.5, 1.5, 2.5],
        64: [0.01, 0.5, 1.5, 2.5],
        128: [0.01, 0.5, 1.5, 2.5]
    }

    K = 5
    seed = 42

    total_cpus = mp.cpu_count()
    cpus_to_use = max(1, math.ceil(total_cpus * 0.7))
    print(f"Total CPUs available: {total_cpus}, using {cpus_to_use} CPUs (70%)")

    manager = mp.Manager()
    lock = manager.Lock()  # Shared lock for file writing

    params_list = []
    for N in N_values:
        e_values = e_values_dict[N]
        full_data = generate_time_series(seed, max(n_values), N)

        for n in n_values:
            T = full_data[:n]
            for e in e_values:
                for alg_func, alg_name in algorithms:
                    for run_id in range(K):
                        params_list.append((alg_func, alg_name, N, T, e, run_id, lock))

    with mp.Pool(processes=cpus_to_use) as pool:
        for _ in pool.imap_unordered(run_experiment, params_list):
            pass

    print("✔️ All experiments completed.")


    

# --- Entry Point ---
if __name__ == "__main__":
    main()