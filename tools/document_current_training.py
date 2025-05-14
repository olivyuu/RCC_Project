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
import logging
import psutil
from typing import Optional, Dict, Any
from tools.debug_utils import DebugLogger

def setup_logging(debug: bool = False) -> DebugLogger:
    """Setup debug logging."""
    logger = DebugLogger("documentation_debug", debug=debug)
    return logger

def validate_documentation(doc: Dict[str, Any], logger: Optional[DebugLogger] = None) -> bool:
    """Validate collected documentation."""
    required_sections = ['metadata', 'code_state']
    missing = [section for section in required_sections if section not in doc]
    
    if missing:
        if logger:
            logger.logger.error(f"Missing required sections: {missing}")
        return False
    
    if 'checkpoints' in doc and not doc['checkpoints']:
        if logger:
            logger.logger.warning("No checkpoint information found")
    
    if 'tensorboard' in doc and not doc['tensorboard']:
        if logger:
            logger.logger.warning("No tensorboard data found")
    
    return True

def collect_checkpoint_info(checkpoint_dir: Path, logger: Optional[DebugLogger] = None) -> dict:
    """Collect information from available checkpoints."""
    checkpoints = {}
    
    # Find all checkpoints
    for cp in checkpoint_dir.glob("*.pth"):
        try:
            # Load checkpoint
            if logger:
                logger.log_memory(f"Before loading checkpoint {cp.name}")
            
            try:
                ckpt = torch.load(cp, map_location='cpu')
            except Exception as e:
                if logger:
                    logger.logger.error(f"Failed to load checkpoint {cp.name}: {e}")
                continue
            
            checkpoints[cp.name] = {
                'epoch': ckpt.get('epoch'),
                'best_val_dice': ckpt.get('best_val_dice'),
                'metrics': ckpt.get('metrics', {}),
                'created': datetime.datetime.fromtimestamp(cp.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                'file_size': cp.stat().st_size / (1024 * 1024),  # Size in MB
                'optimizer_state': {
                    'name': type(ckpt.get('optimizer_state_dict', {})).__name__,
                    'parameters': dict(sorted([
                        (k, v) for k, v in ckpt.get('optimizer_state_dict', {}).items()
                        if isinstance(v, (int, float, str))
                    ]))
                } if 'optimizer_state_dict' in ckpt else None
            }
            
            # Get model architecture if available
            if 'model_state_dict' in ckpt:
                model_keys = list(ckpt['model_state_dict'].keys())
                checkpoints[cp.name]['architecture'] = {
                    'num_parameters': len(model_keys),
                    'layer_names': model_keys[:5] + ['...'] + model_keys[-5:] if len(model_keys) > 10 else model_keys
                }
            
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
                    if any(key in k.lower() for key in [
                        'batch', 'lr', 'epoch', 'size', 'patch', 'seed',
                        'loss', 'optim', 'train', 'val'
                    ])
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
    
    # 4. Try to collect dataset information
    try:
        dataset_info = {
            'train_cases': [],
            'val_cases': []
        }
        
        # Look for case IDs in dataset code or logs
        dataset_file = project_path / "dataset.py"
        if dataset_file.exists():
            with open(dataset_file, 'r') as f:
                content = f.read()
                # Try to find case IDs or dataset split information
                import re
                cases = re.findall(r'case_\d+', content)
                if cases:
                    dataset_info['all_cases'] = sorted(set(cases))
        
        documentation['dataset'] = dataset_info
    except Exception as e:
        print(f"Warning: Could not collect complete dataset information: {e}")
    
    # 5. Create code snapshot
    code_dir = output_path / "code_snapshot"
    code_dir.mkdir(exist_ok=True)
    for py_file in project_path.glob("*.py"):
        if not py_file.name.startswith("__"):
            shutil.copy2(py_file, code_dir / py_file.name)
    
    # Add notes section for manual additions
    documentation['notes'] = {
        'manual_additions_needed': [
            'Add train/val split information if not automatically detected',
            'Specify any special training conditions or observations',
            'Note any training interruptions or adjustments',
            'Document any data preprocessing details'
        ]
    }
    
    # Save documentation
    doc_path = output_path / "training_documentation.yaml"
    with open(doc_path, 'w') as f:
        yaml.safe_dump(documentation, f, default_flow_style=False)
    
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