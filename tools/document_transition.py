"""
Script to document the transition from patch-based to full volume training.
"""
import argparse
from pathlib import Path
import yaml
from datetime import datetime
import json
import torch
import sys

def load_patch_model_info(exp_dir: Path) -> dict:
    """Load information about the patch-trained model."""
    # Find best model checkpoint
    checkpoints = list(exp_dir.glob("best_model_dice_*.pth"))
    if not checkpoints:
        print(f"Error: No best model found in {exp_dir}")
        sys.exit(1)
    
    # Sort by dice score in filename
    best_checkpoint = sorted(
        checkpoints,
        key=lambda x: float(str(x).split("dice_")[-1].replace(".pth", "")),
        reverse=True
    )[0]
    
    # Load checkpoint metadata
    checkpoint = torch.load(best_checkpoint, map_location='cpu')
    
    # Find corresponding config
    config_file = exp_dir / "configs/latest.yaml"
    if not config_file.exists():
        print(f"Warning: No config file found at {config_file}")
        config = None
    else:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    
    return {
        "checkpoint_path": str(best_checkpoint),
        "validation_dice": float(str(best_checkpoint).split("dice_")[-1].replace(".pth", "")),
        "training_date": datetime.fromtimestamp(best_checkpoint.stat().st_mtime).strftime("%Y-%m-%d"),
        "config": config,
        "epochs_trained": checkpoint.get('epoch', 'unknown'),
        "model_state": {
            "best_val_dice": checkpoint.get('best_val_dice', 'unknown'),
            "final_metrics": checkpoint.get('metrics', {})
        }
    }

def load_volume_training_info(exp_dir: Path) -> dict:
    """Load information about the full volume training."""
    config_file = exp_dir / "configs/latest.yaml"
    if not config_file.exists():
        print(f"Warning: No config file found at {config_file}")
        return {}
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Find best volume model
    checkpoints = list(exp_dir.glob("checkpoints/volume_best_dice_*.pth"))
    if not checkpoints:
        print(f"Warning: No volume checkpoints found in {exp_dir}/checkpoints/")
        return {"config": config}
    
    best_checkpoint = sorted(
        checkpoints,
        key=lambda x: float(str(x).split("dice_")[-1].replace(".pth", "")),
        reverse=True
    )[0]
    
    checkpoint = torch.load(best_checkpoint, map_location='cpu')
    
    return {
        "checkpoint_path": str(best_checkpoint),
        "validation_dice": float(str(best_checkpoint).split("dice_")[-1].replace(".pth", "")),
        "config": config,
        "epochs_trained": checkpoint.get('epoch', 'unknown'),
        "model_state": {
            "best_val_dice": checkpoint.get('best_val_dice', 'unknown'),
            "final_metrics": checkpoint.get('metrics', {})
        }
    }

def document_transition(patch_exp: str, volume_exp: str, base_path: str = "experiments"):
    """Document the transition from patch to full volume training."""
    base_dir = Path(base_path)
    patch_dir = base_dir / patch_exp
    volume_dir = base_dir / volume_exp
    
    # Verify directories exist
    if not patch_dir.exists() or not volume_dir.exists():
        print(f"Error: Experiment directories not found")
        sys.exit(1)
    
    # Load information
    patch_info = load_patch_model_info(patch_dir)
    volume_info = load_volume_training_info(volume_dir)
    
    # Create transition document
    transition_doc = {
        "metadata": {
            "documented_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "patch_experiment": patch_exp,
            "volume_experiment": volume_exp
        },
        "patch_training": patch_info,
        "volume_training": volume_info,
        "transition_metrics": {
            "patch_best_dice": patch_info.get("validation_dice"),
            "volume_best_dice": volume_info.get("validation_dice"),
            "dice_improvement": (
                volume_info.get("validation_dice", 0) - 
                patch_info.get("validation_dice", 0)
            )
        }
    }
    
    # Save document
    transitions_dir = base_dir / "transitions"
    transitions_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    doc_path = transitions_dir / f"transition_{patch_exp}_to_{volume_exp}_{timestamp}.yaml"
    
    with open(doc_path, 'w') as f:
        yaml.safe_dump(transition_doc, f, default_flow_style=False)
    
    print(f"Transition documentation saved to: {doc_path}")
    
    # Print summary
    print("\nTransition Summary:")
    print(f"Patch Training Best Dice: {patch_info.get('validation_dice', 'unknown')}")
    print(f"Volume Training Best Dice: {volume_info.get('validation_dice', 'unknown')}")
    if all(x is not None for x in [patch_info.get('validation_dice'), volume_info.get('validation_dice')]):
        print(f"Dice Improvement: {transition_doc['transition_metrics']['dice_improvement']:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Document patch to volume transition")
    parser.add_argument("--patch_exp", required=True, help="Patch training experiment name")
    parser.add_argument("--volume_exp", required=True, help="Volume training experiment name")
    args = parser.parse_args()
    
    document_transition(args.patch_exp, args.volume_exp)

if __name__ == "__main__":
    main()