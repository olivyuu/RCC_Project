"""
Script to save and track experiment configurations.
"""
import argparse
from pathlib import Path
import yaml
from datetime import datetime
import sys
import shutil

def load_config(config_path: str) -> dict:
    """Load configuration from file."""
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Error loading config file: {e}")
            sys.exit(1)
    return config

def save_config(exp_name: str, config: dict, base_path: str = "experiments"):
    """Save configuration with metadata to experiment directory."""
    exp_dir = Path(base_path) / exp_name
    config_dir = exp_dir / "configs"
    
    if not config_dir.exists():
        print(f"Error: Experiment directory not found: {exp_dir}")
        print("Please create experiment first using create_experiment.py")
        sys.exit(1)
    
    # Add metadata to config
    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_name": exp_name
    }
    
    config_with_meta = {
        "metadata": metadata,
        "configuration": config
    }
    
    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = config_dir / f"config_{timestamp}.yaml"
    
    with open(config_path, 'w') as f:
        yaml.safe_dump(config_with_meta, f, default_flow_style=False)
    
    # Also save as latest.yaml
    latest_path = config_dir / "latest.yaml"
    shutil.copy(config_path, latest_path)
    
    print(f"Saved configuration to: {config_path}")
    print(f"Updated latest config at: {latest_path}")
    return config_path

def main():
    parser = argparse.ArgumentParser(description="Save experiment configuration")
    parser.add_argument("--exp", required=True, help="Experiment name")
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()
    
    # Load and save config
    config = load_config(args.config)
    save_config(args.exp, config)

if __name__ == "__main__":
    main()