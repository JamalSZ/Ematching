import multiprocessing as mp
import math
from tqdm import tqdm
import time
import tracemalloc
import csv
import numpy as np
import pandas as pd


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
    alg, alg_name, dataset, series, e, run_id = params
    n = len(series)

    tracemalloc.start()
    start_time = time.perf_counter()
    alg(series, e, n=n)
    end_time = time.perf_counter()
    current, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    rt = end_time - start_time
    mem = peak_mem / (1024 * 1024)  # Convert bytes to MB

    filename = f"Exp_results/Exp_realworld/{alg_name}_{dataset}.csv"
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([alg_name, len(series), e, rt, mem, run_id])

    return [alg_name, len(series), e, rt, mem, run_id]

# --- Time Series Generator ---
def get_time_series(path,n,col=None):
    print(path)
    
    if col == None:
        df = pd.read_csv(path,header=None)
        T = df[0].values[:n]
    else:
        df = pd.read_csv(path)
        T= df[col].values[:n]
    T = list(T)
    return T

# --- Main Execution ---

def main():
    algorithms = [
        (algorithm_bf, "BF"),
        (algorithm_rnlj, "RNLJ"),
        (algorithm_ifi, "IFI"),
        (algorithm_iesj, "IESJ"),
        (algorithm_fit, "FIT")
    ]

    length = {
        "temp": [2000],
        "stock": [4000],
        "etth1": [2000]
        }
    datasets = ["temp","etth1","stock"]
    paths = {
        "temp":"Datasets/temperature.csv",
        "stock":"Datasets/Stock_Open.csv",
        "etth1":"Datasets/ETTh1.csv"
    }
    e_values_dict = {
        "temp": [0.1,0.2, 0.3,0.4,0.5],
        "stock": [0.1,2,4,6,8],
        "etth1": [0.1,0.2, 0.3,0.4,0.5]
    }
    columns = {
        "temp":"HT",
        "stock":"OP",
        "etth1":"OT"
    }

    K = 1  # Repetitions
    seed = 42
    
    # Calculate number of CPUs to use (70% of total)
    total_cpus = mp.cpu_count()
    cpus_to_use = max(1, math.ceil(total_cpus * 0.7))
    print(f"Total CPUs available: {total_cpus}, using {cpus_to_use} CPUs (7%)")

    params_list = []
    for dataset in datasets:
        e_values = e_values_dict[dataset]
        n_values = length[dataset]
        col = columns[dataset]
        full_data = get_time_series(paths[dataset], max(n_values),col)

        for n in n_values:
            T = full_data[:n]
            for e in e_values:
                params_list.append([
                    (alg_func, alg_name, dataset, T, e, run_id)
                    for alg_func, alg_name in algorithms
                    for run_id in range(K)
                ])

                #print(f"Running experiments for dataset={dataset}, n={n}, e={e}...")
                #print(params_list[-1])
    
    for params in params_list:
        with mp.Pool(processes=cpus_to_use) as pool:  # Use the calculated number of CPUs
            for _ in pool.imap_unordered(run_experiment, params):
                pass

    print(f"✔️ All experiments for completed.")
    

    

# --- Entry Point ---
if __name__ == "__main__":
    main()