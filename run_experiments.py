import os
import subprocess
import sys
from time import time
from datetime import timedelta

def run_experiment(exp, exp_plot):
    """Run a single experiment and its corresponding plotting script"""
    exp_script = f"pyExp/{exp}.py"
    plot_script = f"pyPlots/plot_{exp_plot}.py"
    
    # Check if files exist
    if not os.path.exists(exp_script):
        print(f"Error: Experiment script {exp_script} not found!")
        return False
    if not os.path.exists(plot_script):
        print(f"Error: Plotting script {plot_script} not found!")
        return False

    # Create output directory based on experiment name (e.g., exp_realworld → Exp#realworld)
    exp_suffix = exp.replace("exp_", "")
    output_dir = os.path.join("Exp_results", f"Exp_{exp_suffix}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Running Experiment {exp}")
    print(f"{'='*50}")
    
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Run experiment
    print(f"\nExecuting experiment script: {exp_script}")
    start_time = time()
    try:
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
    experiment = ["exp_realworld", "exp_max_n", "exp_edge"]
    experiment_plot = ["realworld", "max_n", "edge"]

    for exp_num in range(4):
        exp = experiment[exp_num]
        exp_plot = experiment_plot[exp_num]
        success = run_experiment(exp, exp_plot)
        if not success:
            print(f"Stopping pipeline due to failure in Experiment {exp_num}")
            continue
    
    print("\nExperiment and plotting pipeline completed")

if __name__ == "__main__":
    main()
