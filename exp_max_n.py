import numpy as np
import csv
import os
import time
import random
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import List, Dict, Tuple, Callable

# --- Setup Logging (File Only) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("experiment.log")]  # No StreamHandler
)
logger = logging.getLogger(__name__)

# --- Algorithm Imports ---
from algs.iesj import *
from algs.ifi import *
from algs.bf import *
from algs.rnlj import *
from algs.fit import *



# --- Constants ---
SEED = 42
DEFAULT_N = 500_000  # Base time series length
DEFAULT_N_MAX = 128  # Max value in series
TIME_LIMITS = [1,1.5,2]#, 4, 6, 8, 10]  # Seconds
E_VALUES = [0.01, 0.02, 0.03] #0.51, 1.51, 2.51]
OUTPUT_DIR = "Exp_Results/Exp_BreakPoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)


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
def run_algorithm(
    algorithm: Callable,
    data: List[int],
    e: float,
    time_limit: float
) -> Tuple[float, bool]:
    """Execute an algorithm with timeout and return runtime + success flag."""
    start_time = time.time()
    try:
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(algorithm, data, e)
            result = future.result(timeout=time_limit)
            runtime = time.time() - start_time
            return runtime, True
    except TimeoutError:
        future.cancel()
        return time_limit + 1, False
    except Exception as ex:
        logger.error(f"Algorithm {algorithm.__name__} failed: {ex}")
        return time_limit + 1, False

# --- Core Experiment Logic ---
def find_max_data_size(
    algorithm: Callable,
    data: List[int],
    e: float,
    time_limit: float,
    min_size: int = 10,
    max_size: int = None,
    step_size: int = 4000,
    tolerance: int = 100
) -> int:
    """Binary search to find max input size within time limit."""
    max_size = max_size or len(data)
    low, high = min_size, min_size
    best_size = min_size

    # Phase 1: Exponential search to find upper bound
    while high <= max_size:
        runtime, success = run_algorithm(algorithm, data[:high], e, time_limit)
        if not success:
            break
        best_size = high
        low = high
        prev_n = high
        high += step_size


    high = min(high, max_size)
    low = prev_n
    best_size = prev_n

    # Phase 2: Binary search for precise limit
    while low <= high:
        mid = (low + high) // 2
        runtime, success = run_algorithm(algorithm, data[:mid], e, time_limit)

        if success:
            best_size = mid
            low = mid + 1
        else:
            high = mid - 1
            best_size = mid-1

        if abs(prev_n - best_size) <= tolerance:
            best_size = (prev_n +best_size) // 2
            break
        prev_n = best_size

    return best_size

def generate_time_series(seed: int, n: int, max_val: int) -> List[int]:
    """Generate reproducible random time series."""
    random.seed(seed)
    return [random.randint(0, max_val) for _ in range(n)]

def run_single_experiment(
    algorithm,
    name,
    data,
    e,
    time_limit
) -> Dict[str, float]:
    """Run one experiment and return results."""
    max_size = find_max_data_size(algorithm, data, e, time_limit)
    return {
        "algorithm": name,
        "e": e,
        "time_limit": time_limit,
        "max_size": max_size
    }

def save_results(results: List[Dict[str, float]], e: float):
    """Save results to a CSV file for a specific e."""
    output_file = os.path.join(OUTPUT_DIR, f"exp_bp_{e}.csv")
    headers = ["algorithm", "time_limit", "max_size","e"]
    write_header = not os.path.exists(output_file)

    with open(output_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if write_header:
            writer.writeheader()
        writer.writerows(results)

# --- Main Execution ---
def main():
    algorithms = {
        "BF": algorithm_bf,
        "RNLJ": algorithm_rnlj,
        "IFI": algorithm_ifi,
        "IESJ": algorithm_iesj,
        "FIT": algorithm_fit
    }

    data = generate_time_series(SEED, DEFAULT_N, DEFAULT_N_MAX)

    for e in E_VALUES:
        logger.info(f"Starting experiments for e={e}")
        results = []

        with ProcessPoolExecutor() as executor:
            futures = []
            for name, algo in algorithms.items():
                for t_limit in TIME_LIMITS:
                    futures.append(
                        executor.submit(
                            run_single_experiment,
                            algo,
                            name,
                            data,
                            e,
                            t_limit
                        )
                    )

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(
                        f"Completed {result['algorithm']} @ e={e}, "
                        f"t={result['time_limit']}: max_size={result['max_size']}"
                    )
                except Exception as ex:
                    logger.error(f"Experiment failed: {ex}")

        save_results(results, e)

if __name__ == "__main__":
    main()