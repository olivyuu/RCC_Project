import argparse
from pathlib import Path
import torch
import random
import numpy as np
from models.segmentation import SegmentationModel, SegmentationConfig
from models.segmentation.trainer import SegmentationTrainer
from models.segmentation.dataset import SegmentationDataset

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print(f"Set random seed to {seed}")

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
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--resume", action="store_true",
                       help="Resume training from latest checkpoint")
    parser.add_argument("--preprocess", type=str, choices=["true", "false"],
                       help="Whether to preprocess data or use existing preprocessed files")
    args = parser.parse_args()

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Set random seed
    set_seed(args.seed)

    # Initialize configuration
    config = SegmentationConfig()
    
    # Update config with command line arguments
    config.data_dir = args.data_dir
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.initial_lr = args.lr
    config.seed = args.seed
    config.resume_training = args.resume
    config.checkpoint_dir = checkpoints_dir
    config.log_dir = logs_dir
    config.experiment_name = f"segmentation_bs{args.batch_size}_lr{args.lr}"
    config.preprocess = True if args.preprocess is None else args.preprocess.lower() == "true"

    # Create trainer
    trainer = SegmentationTrainer(config)

    # Start training
    try:
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