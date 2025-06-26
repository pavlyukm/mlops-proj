#!/usr/bin/env python3
"""Initialize MLflow directories and experiment"""
import os
import yaml

def init_mlflow():
    # Create mlruns directory structure
    mlruns_dir = "mlruns"
    if not os.path.exists(mlruns_dir):
        os.makedirs(mlruns_dir)
    
    # Create default experiment (0)
    default_exp_dir = os.path.join(mlruns_dir, "0")
    if not os.path.exists(default_exp_dir):
        os.makedirs(default_exp_dir)
        
        # Create meta.yaml for default experiment
        meta_content = {
            'artifact_location': 'file:///mlflow/artifacts/0',
            'creation_time': 1640995200000,
            'experiment_id': '0',
            'last_update_time': 1640995200000,
            'lifecycle_stage': 'active',
            'name': 'Default'
        }
        
        with open(os.path.join(default_exp_dir, 'meta.yaml'), 'w') as f:
            yaml.dump(meta_content, f)
    
    print("MLflow directories initialized successfully")

if __name__ == "__main__":
    init_mlflow()