import os
import sys
import torch
import argparse
from pathlib import Path

from models.segmentation.patch_trainer import PatchSegmentationTrainer
from models.segmentation.config import SegmentationConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Train tumor segmentation model using patches')
    
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to preprocessed volume data')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save checkpoints and logs')
    parser.add_argument('--patch_size', type=str, default='64,128,128',
                      help='Patch dimensions as D,H,W (default: 64,128,128)')
    parser.add_argument('--batch_size', type=int, default=2,
                      help='Batch size for patch training (default: 2)')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of epochs to train (default: 100)')
    parser.add_argument('--tumor_prob', type=float, default=0.7,
                      help='Probability of sampling tumor-centered patches (default: 0.7)')
    parser.add_argument('--pos_weight', type=float, default=10.0,
                      help='Positive class weight in loss function (default: 10.0)')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                      help='Initial learning rate (default: 5e-5)')
    parser.add_argument('--resume', action='store_true',
                      help='Resume training from last checkpoint')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug output')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Configure deterministic training
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)
    
    print("Training will use non-deterministic operations with fixed random seeds")
    print(f"Using PyTorch version {torch.__version__}")
    
    # Create config
    config = SegmentationConfig()
    config.data_dir = Path(args.data_dir)
    config.output_dir = Path(args.output_dir)
    config.checkpoint_dir = config.output_dir / 'checkpoints'
    config.log_dir = config.output_dir / 'logs'
    config.num_epochs = args.epochs
    config.resume_training = args.resume
    
    # Create output directories
    config.output_dir.mkdir(exist_ok=True, parents=True)
    config.checkpoint_dir.mkdir(exist_ok=True)
    config.log_dir.mkdir(exist_ok=True)
    
    # Create trainer
    trainer = PatchSegmentationTrainer(config)
    
    # Override trainer settings from arguments
    patch_dims = [int(x) for x in args.patch_size.split(',')]
    trainer.patch_size = tuple(patch_dims)
    trainer.batch_size = args.batch_size
    
    # Update dataset settings
    trainer.tumor_only_prob = args.tumor_prob
    
    # Update optimizer settings
    trainer.optimizer.param_groups[0]['lr'] = args.learning_rate
    
    # Update loss settings
    trainer.criterion.pos_weight = torch.tensor([args.pos_weight])
    
    # Enable debug output if requested
    if args.debug:
        print("\nDebug mode enabled")
        print(f"Patch size: {trainer.patch_size}")
        print(f"Batch size: {trainer.batch_size}")
        print(f"Tumor sampling probability: {trainer.tumor_only_prob}")
        print(f"Positive class weight: {args.pos_weight}")
        print(f"Learning rate: {args.learning_rate}")
    
    try:
        # Start training
        trainer.train(args.data_dir)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

if __name__ == '__main__':
    main()