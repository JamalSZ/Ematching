import os
import subprocess
import sys
from time import time
from datetime import timedelta

def run_experiment(exp, exp_plot):
    """Run a single experiment and its corresponding plotting script"""
    exp_script = f"pyExp/{exp}.py"  # Modified to match your actual filename
    plot_script = f"pyPlots/plot_{exp_plot}.py"
    
    # Check if files exist
    if not os.path.exists(exp_script):
        print(f"Error: Experiment script {exp_script} not found!")
        return False
    if not os.path.exists(plot_script):
        print(f"Error: Plotting script {plot_script} not found!")
        return False
    
    print(f"\n{'='*50}")
    print(f"Running Experiment {exp}")
    print(f"{'='*50}")
    
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Run experiment
    print(f"\nExecuting experiment script: {exp_script}")
    start_time = time()
    try:
        # Add project root to PYTHONPATH before running
        env = os.environ.copy()
        env['PYTHONPATH'] = project_root + (os.pathsep + env['PYTHONPATH']) if 'PYTHONPATH' in env else project_root
        
        subprocess.run([sys.executable, exp_script], check=True, env=env)
        exp_time = timedelta(seconds=time()-start_time)
        print(f"Experiment {exp} completed successfully in {exp_time}")
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment {exp}: {e}")
        return False
    
    # Run plotting
    print(f"\nExecuting plotting script: {plot_script}")
    start_time = time()
    try:
        subprocess.run([sys.executable, plot_script], check=True)
        plot_time = timedelta(seconds=time()-start_time)
        print(f"Plotting for Experiment {exp} completed successfully in {plot_time}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running plotting for experiment {exp}: {e}")
        return False


def main():
    print("Starting experiment and plotting pipeline")
    print(f"Python executable: {sys.executable}")
    experiment = ["exp_realworld", "exp_edge", "exp_max_n", "exp_synthetic"]
    experiment_plot = ["realworld", "edge", "max_n", "synthetic"]
    # Run experiments 1 through 4 in sequence
    for exp_num in range(0, 4):
        exp = experiment[exp_num]
        exp_plot = experiment_plot[exp_num]
        success = run_experiment(exp,exp_plot)
        if not success:
            print(f"Stopping pipeline due to failure in Experiment {exp_num}")
            continue
    
    print("\nExperiment and plotting pipeline completed")

if __name__ == "__main__":
    main()