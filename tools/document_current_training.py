"""
Script to retroactively document an existing training session.
Collects and organizes all available information for reproducibility.
"""
import argparse
from pathlib import Path
import yaml
import json
import datetime
import shutil
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import sys

def convert_tensor_to_python(obj):
    """Convert PyTorch tensors to Python native types."""
    if torch.is_tensor(obj):
        return float(obj.item()) if obj.numel() == 1 else obj.cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {key: convert_tensor_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_tensor_to_python(item) for item in obj]
    return obj

def collect_checkpoint_info(checkpoint_dir: Path) -> dict:
    """Collect information from available checkpoints."""
    checkpoints = {}
    
    # Find all checkpoints
    for cp in checkpoint_dir.glob("*.pth"):
        try:
            # Load checkpoint
            ckpt = torch.load(cp, map_location='cpu')
            
            # Convert tensors to Python types
            ckpt_data = {
                'epoch': convert_tensor_to_python(ckpt.get('epoch')),
                'best_val_dice': convert_tensor_to_python(ckpt.get('best_val_dice')),
                'metrics': convert_tensor_to_python(ckpt.get('metrics', {})),
                'created': datetime.datetime.fromtimestamp(cp.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                'file_size': cp.stat().st_size / (1024 * 1024)  # Size in MB
            }
            checkpoints[cp.name] = ckpt_data
            
            del ckpt  # Free memory
            
        except Exception as e:
            print(f"Warning: Could not load checkpoint {cp.name}: {e}")
    
    return checkpoints

def collect_tensorboard_info(runs_dir: Path) -> dict:
    """Extract information from tensorboard logs."""
    tensorboard_data = {}
    
    # Find all event files
    for run_dir in runs_dir.glob("*"):
        if not run_dir.is_dir():
            continue
            
        event_files = list(run_dir.glob("events.out.tfevents.*"))
        if not event_files:
            continue
            
        try:
            # Load events
            acc = EventAccumulator(str(run_dir))
            acc.Reload()
            
            # Extract scalars
            scalars = {}
            for tag in acc.Tags()['scalars']:
                events = acc.Scalars(tag)
                scalars[tag] = {
                    'min': min(e.value for e in events),
                    'max': max(e.value for e in events),
                    'last': events[-1].value,
                    'steps': len(events)
                }
                
            tensorboard_data[run_dir.name] = {
                'scalars': scalars,
                'first_event': datetime.datetime.fromtimestamp(event_files[0].stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                'last_event': datetime.datetime.fromtimestamp(event_files[-1].stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            print(f"Warning: Could not process tensorboard logs in {run_dir}: {e}")
    
    return tensorboard_data

def collect_code_state(project_dir: Path) -> dict:
    """Collect information about the code state."""
    code_info = {
        'files': {},
        'key_parameters': {}
    }
    
    # Collect info about key Python files
    for py_file in project_dir.glob("*.py"):
        if py_file.name.startswith("__"):
            continue
            
        try:
            with open(py_file, 'r') as f:
                content = f.read()
                
            code_info['files'][py_file.name] = {
                'size': len(content),
                'modified': datetime.datetime.fromtimestamp(py_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                'lines': len(content.splitlines())
            }
            
            # Extract key parameters from config.py
            if py_file.name == 'config.py':
                import re
                # Look for common parameter patterns
                params = re.findall(r'(\w+)\s*[=:]\s*([^#\n]+)', content)
                code_info['key_parameters'].update({
                    k.strip(): v.strip()
                    for k, v in params 
                    if any(key in k.lower() for key in ['batch', 'lr', 'epoch', 'size', 'patch', 'seed'])
                })
                
        except Exception as e:
            print(f"Warning: Could not process {py_file.name}: {e}")
    
    return code_info

def document_current_training(project_dir: str, output_dir: str):
    """Document current training session."""
    project_path = Path(project_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    documentation = {
        'metadata': {
            'documented_at': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'project_directory': str(project_path)
        }
    }
    
    # 1. Collect checkpoint information
    print("Collecting checkpoint information...")
    checkpoints_dir = project_path / "checkpoints"
    if checkpoints_dir.exists():
        documentation['checkpoints'] = collect_checkpoint_info(checkpoints_dir)
        
        # Copy best checkpoint to experiments directory
        best_checkpoints = [
            cp for cp in checkpoints_dir.glob("best_model_*.pth")
        ]
        if best_checkpoints:
            best_cp = max(best_checkpoints, key=lambda p: p.stat().st_mtime)
            shutil.copy2(best_cp, output_path / best_cp.name)
    
    # 2. Collect tensorboard information
    print("Collecting training logs...")
    runs_dir = project_path / "runs"
    if runs_dir.exists():
        documentation['tensorboard'] = collect_tensorboard_info(runs_dir)
        
        # Copy tensorboard logs
        tb_dir = output_path / "tensorboard"
        tb_dir.mkdir(exist_ok=True)
        for item in runs_dir.glob("*"):
            if item.is_dir():
                shutil.copytree(item, tb_dir / item.name, dirs_exist_ok=True)
    
    # 3. Collect code state
    print("Documenting code state...")
    documentation['code_state'] = collect_code_state(project_path)
    
    # Save documentation with safeguards for PyTorch tensors
    doc_path = output_path / "training_documentation.yaml"
    try:
        with open(doc_path, 'w') as f:
            yaml.safe_dump(convert_tensor_to_python(documentation), f, default_flow_style=False)
    except Exception as e:
        print(f"Error saving documentation: {e}")
        # Try JSON as fallback
        json_path = output_path / "training_documentation.json"
        with open(json_path, 'w') as f:
            json.dump(convert_tensor_to_python(documentation), f, indent=2)
        print(f"Documentation saved as JSON instead at: {json_path}")
        return
    
    print(f"\nTraining documentation saved to: {doc_path}")
    print("\nCollected Information:")
    print(f"- Checkpoints: {len(documentation.get('checkpoints', {}))}")
    print(f"- Code Files: {len(documentation['code_state']['files'])}")
    if 'tensorboard' in documentation:
        print(f"- Training Runs: {len(documentation['tensorboard'])}")
    
    print("\nNext Steps:")
    print("1. Review training_documentation.yaml for completeness")
    print("2. Note the best model checkpoint and its performance")
    print("3. Use this information when starting full volume training")
    print("4. Consider adding any missing information manually to the documentation")

def main():
    parser = argparse.ArgumentParser(description="Document current training session")
    parser.add_argument("--project_dir", default=".", help="Project directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for documentation")
    args = parser.parse_args()
    
    document_current_training(args.project_dir, args.output_dir)

if __name__ == "__main__":
    main()