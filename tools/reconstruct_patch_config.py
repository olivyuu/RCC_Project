"""
Script to reconstruct patch training configuration from all available sources.
"""
import argparse
from pathlib import Path
import yaml
import torch
import datetime
import re
import ast
import json
from typing import Dict, Any

def extract_config_from_code(config_file: Path, trainer_file: Path) -> Dict[str, Any]:
    """Extract configuration parameters from config.py and trainer.py."""
    config = {}
    
    # Extract from config.py
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Look for parameter patterns
    params = re.findall(r'(\w+)\s*[=:]\s*([^#\n]+)', content)
    for k, v in params:
        if any(key in k.lower() for key in [
            'batch', 'lr', 'epoch', 'size', 'patch', 'seed', 
            'features', 'channel', 'weight', 'chunk', 'stride',
            'patience', 'factor', 'amp', 'augment', 'loss'
        ]):
            try:
                # Try to evaluate the value if it's a Python expression
                config[k] = ast.literal_eval(v.strip())
            except:
                config[k] = v.strip()
    
    # Extract from trainer.py
    with open(trainer_file, 'r') as f:
        content = f.read()
    
    # Look for training settings
    if 'optimizer =' in content:
        optimizer_match = re.search(r'optimizer\s*=\s*(\w+)', content)
        if optimizer_match:
            config['optimizer'] = optimizer_match.group(1)
    
    if 'criterion =' in content:
        loss_match = re.search(r'criterion\s*=\s*(\w+)', content)
        if loss_match:
            config['loss_function'] = loss_match.group(1)
    
    return config

def extract_config_from_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Extract configuration information from the checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    config = {}
    
    # Model architecture
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        # Input/output channels
        first_conv = [k for k in state_dict.keys() if 'conv1.weight' in k][0]
        final_layer = [k for k in state_dict.keys() if 'final' in k and 'weight' in k][0]
        config['model_architecture'] = {
            'in_channels': state_dict[first_conv].shape[1],
            'out_channels': state_dict[final_layer].shape[0],
            'first_layer_features': state_dict[first_conv].shape[0],
            'layer_structure': [
                k.split('.')[0] for k in state_dict.keys() 
                if k.endswith('weight') and 'conv' in k
            ]
        }
    
    # Training state
    config['training_state'] = {
        'epochs_completed': ckpt.get('epoch'),
        'best_val_dice': float(ckpt['best_val_dice'].item()) if 'best_val_dice' in ckpt else None,
        'final_metrics': {
            k: float(v.item()) if torch.is_tensor(v) else v
            for k, v in ckpt.get('metrics', {}).items()
        }
    }
    
    return config

def extract_config_from_logs(runs_dir: Path) -> Dict[str, Any]:
    """Extract training information from tensorboard logs."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    
    log_data = {}
    for run_dir in runs_dir.glob("*"):
        if not run_dir.is_dir():
            continue
        
        acc = EventAccumulator(str(run_dir))
        acc.Reload()
        
        # Get scalar summaries
        scalars = {}
        for tag in acc.Tags()['scalars']:
            events = acc.Scalars(tag)
            scalars[tag] = {
                'min': min(e.value for e in events),
                'max': max(e.value for e in events),
                'final': events[-1].value,
                'steps': len(events)
            }
        
        log_data[run_dir.name] = scalars
    
    return log_data

def reconstruct_patch_config(exp_dir: str):
    """Reconstruct patch training configuration."""
    exp_path = Path(exp_dir)
    project_root = exp_path.parent.parent
    
    reconstructed_config = {
        "metadata": {
            "reconstructed_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_dir": str(exp_path),
            "best_model": None
        },
        "sources": [],
        "configuration": {},
        "training_results": {}
    }
    
    # 1. Code configuration
    try:
        config_file = project_root / "config.py"
        trainer_file = project_root / "trainer.py"
        if config_file.exists() and trainer_file.exists():
            code_config = extract_config_from_code(config_file, trainer_file)
            reconstructed_config['configuration'].update(code_config)
            reconstructed_config['sources'].append("config.py, trainer.py")
    except Exception as e:
        print(f"Warning: Error extracting code config: {e}")
    
    # 2. Checkpoint configuration
    try:
        checkpoints = list(exp_path.glob("best_model_dice_*.pth"))
        if checkpoints:
            best_cp = max(
                checkpoints,
                key=lambda p: float(str(p).split("dice_")[-1].replace(".pth", ""))
            )
            checkpoint_config = extract_config_from_checkpoint(best_cp)
            reconstructed_config['configuration'].update(checkpoint_config)
            reconstructed_config['metadata']['best_model'] = best_cp.name
            reconstructed_config['sources'].append(f"checkpoint: {best_cp.name}")
    except Exception as e:
        print(f"Warning: Error extracting checkpoint config: {e}")
    
    # 3. Training logs
    try:
        runs_dir = project_root / "runs"
        if runs_dir.exists():
            log_config = extract_config_from_logs(runs_dir)
            reconstructed_config['training_results']['tensorboard_logs'] = log_config
            reconstructed_config['sources'].append("tensorboard logs")
    except Exception as e:
        print(f"Warning: Error extracting training logs: {e}")
    
    # Save reconstructed config
    config_dir = exp_path / "configs"
    config_dir.mkdir(exist_ok=True)
    
    config_path = config_dir / "reconstructed_config.yaml"
    with open(config_path, 'w') as f:
        yaml.safe_dump(reconstructed_config, f, default_flow_style=False)
    
    print(f"\nReconstructed configuration saved to: {config_path}")
    print("\nSources used:", ", ".join(reconstructed_config['sources']))
    if reconstructed_config['metadata']['best_model']:
        print(f"\nBest model: {reconstructed_config['metadata']['best_model']}")
    
    # Also save as JSON for better formatting
    with open(config_dir / "reconstructed_config.json", 'w') as f:
        json.dump(reconstructed_config, f, indent=2)
    
    print("\nKey parameters reconstructed:")
    for k, v in reconstructed_config['configuration'].items():
        if isinstance(v, dict):
            print(f"{k}:")
            for sub_k, sub_v in v.items():
                print(f"  {sub_k}: {sub_v}")
        else:
            print(f"{k}: {v}")

def main():
    parser = argparse.ArgumentParser(description="Reconstruct patch training configuration")
    parser.add_argument("--exp_dir", required=True, help="Path to patch experiment directory")
    args = parser.parse_args()
    
    reconstruct_patch_config(args.exp_dir)

if __name__ == "__main__":
    main()