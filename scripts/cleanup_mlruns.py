#!/usr/bin/env python3
"""Clean up malformed MLflow runs"""
import os
import shutil
import yaml

def cleanup_mlruns(mlruns_dir="mlruns"):
    """Remove malformed runs that are missing meta.yaml files"""
    if not os.path.exists(mlruns_dir):
        print(f"Directory {mlruns_dir} does not exist")
        return
    
    cleaned = 0
    for experiment_id in os.listdir(mlruns_dir):
        exp_path = os.path.join(mlruns_dir, experiment_id)
        if not os.path.isdir(exp_path) or experiment_id == "0":
            continue
            
        # Check experiment meta.yaml
        exp_meta = os.path.join(exp_path, "meta.yaml")
        if not os.path.exists(exp_meta):
            print(f"Skipping experiment {experiment_id} - no meta.yaml")
            continue
            
        # Check runs
        for run_id in os.listdir(exp_path):
            run_path = os.path.join(exp_path, run_id)
            if not os.path.isdir(run_path) or run_id == "meta.yaml":
                continue
                
            meta_path = os.path.join(run_path, "meta.yaml")
            if not os.path.exists(meta_path):
                print(f"Removing malformed run {run_id}")
                shutil.rmtree(run_path)
                cleaned += 1
            else:
                # Check if meta.yaml is valid
                try:
                    with open(meta_path, 'r') as f:
                        yaml.safe_load(f)
                except:
                    print(f"Removing run with corrupted meta.yaml: {run_id}")
                    shutil.rmtree(run_path)
                    cleaned += 1
    
    print(f"Cleaned up {cleaned} malformed runs")

if __name__ == "__main__":
    cleanup_mlruns()