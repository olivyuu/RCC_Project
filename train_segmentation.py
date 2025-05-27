import argparse
from pathlib import Path
import torch
import random
import numpy as np
from models.segmentation import SegmentationModel, SegmentationConfig
from models.segmentation.trainer import SegmentationTrainer
from models.segmentation.dataset import SegmentationDataset
import pkg_resources
import os

def check_versions():
    """Verify that installed package versions match requirements"""
    required = {
        "numpy": "1.24.3",
        "nibabel": "5.1.0",
        "totalsegmentator": "1.5.5",
        "nnunetv2": "2.2.0"
    }
    
    # Check PyTorch version - allow 2.x versions
    torch_version = pkg_resources.get_distribution("torch").version
    if not torch_version.startswith("2."):
        print(f"Error: PyTorch major version mismatch. Have {torch_version}, need version 2.x")
        raise ValueError("PyTorch version must be 2.x for reproducibility")
    else:
        print(f"Using PyTorch version {torch_version}")
    
    for package, version in required.items():
        try:
            installed = pkg_resources.get_distribution(package).version
            if installed != version:
                print(f"Warning: {package} version mismatch. Required: {version}, Installed: {installed}")
        except pkg_resources.DistributionNotFound:
            print(f"Error: Required package {package} is not installed")
            raise

def set_reproducibility():
    """Set up complete reproducibility"""
    # Python RNG
    random.seed(42)
    # Numpy RNG
    np.random.seed(42)
    # PyTorch RNGs
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # Force deterministic operations
    os.environ['PYTHONHASHSEED'] = '42'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Needed for CUDA >= 10.2
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    
    print("Deterministic mode enabled for reproducibility")

def main():
    parser = argparse.ArgumentParser(description="Train tumor segmentation model")
    parser.add_argument("--data_dir", type=str, required=True, 
                       help="Path to KiTS23 dataset directory")
    parser.add_argument("--output_dir", type=str, default="outputs/segmentation",
                       help="Directory to save model outputs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=2e-4,
                       help="Initial learning rate")
    parser.add_argument("--resume", action="store_true",
                       help="Resume training from latest checkpoint")
    parser.add_argument("--preprocess", type=str, choices=["true", "false"],
                       help="Whether to preprocess data or use existing preprocessed files")
    args = parser.parse_args()

    # Check package versions
    check_versions()
    
    # Set up reproducibility
    set_reproducibility()

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Save configuration
    config = SegmentationConfig()
    config.data_dir = args.data_dir
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.initial_lr = args.lr
    config.resume_training = args.resume
    config.checkpoint_dir = checkpoints_dir
    config.log_dir = logs_dir
    config.experiment_name = f"segmentation_bs{args.batch_size}_lr{args.lr}"
    config.preprocess = True if args.preprocess is None else args.preprocess.lower() == "true"

    # Save all configuration and environment details
    with open(output_dir / "experiment_config.txt", "w") as f:
        f.write(f"PyTorch Version: {torch.__version__}\n")
        f.write(f"CUDA Available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"CUDA Version: {torch.version.cuda}\n")
            f.write(f"GPU Device: {torch.cuda.get_device_name(0)}\n")
        f.write("\nConfiguration:\n")
        for key, value in vars(config).items():
            f.write(f"{key}: {value}\n")

    # Create trainer and start training
    try:
        trainer = SegmentationTrainer(config)
        trainer.train(config.data_dir)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        trainer._save_checkpoint(
            trainer.current_epoch,
            {'dice': trainer.best_val_dice},
            is_best=False
        )
        print("Checkpoint saved. Exiting...")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise
    finally:
        if hasattr(trainer, 'writer'):
            trainer.writer.close()

if __name__ == "__main__":
    main()
