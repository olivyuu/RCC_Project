"""
Script to create a new experiment directory with proper structure.
"""
import argparse
from pathlib import Path
import shutil
import yaml
from datetime import datetime

def create_experiment_structure(name: str, base_path: str = "experiments"):
    """Create a new experiment directory with standardized structure."""
    base_dir = Path(base_path)
    exp_dir = base_dir / name
    
    # Create directory structure
    dirs = [
        "configs",
        "checkpoints",
        "logs",
        "splits"
    ]
    
    for d in dirs:
        (exp_dir / d).mkdir(parents=True, exist_ok=True)
    
    # Create experiment info file
    info = {
        "name": name,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "directories": dirs,
        "status": "initialized"
    }
    
    with open(exp_dir / "experiment_info.yaml", "w") as f:
        yaml.safe_dump(info, f)
    
    print(f"Created experiment directory structure at: {exp_dir}")
    print("Created directories:")
    for d in dirs:
        print(f"  - {d}/")

def main():
    parser = argparse.ArgumentParser(description="Create new experiment directory structure")
    parser.add_argument("--name", required=True, help="Name of the experiment")
    args = parser.parse_args()
    
    create_experiment_structure(args.name)

if __name__ == "__main__":
    main()