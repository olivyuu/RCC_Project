import os
import sys
import torch
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # For servers without display

from models.segmentation.patch_trainer import PatchSegmentationTrainer
from models.segmentation.config import SegmentationConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Train tumor segmentation model using patches')
    
    # Data paths
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to preprocessed volume data')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save checkpoints and logs')
    
    # Training settings
    parser.add_argument('--patch_size', type=str, default='64,128,128',
                      help='Patch dimensions as D,H,W (default: 64,128,128)')
    parser.add_argument('--batch_size', type=int, default=2,
                      help='Batch size for patch training (default: 2)')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of epochs to train (default: 100)')
    parser.add_argument('--tumor_prob', type=float, default=0.7,
                      help='Probability of sampling tumor-centered patches (default: 0.7)')
    
    # Loss function settings
    parser.add_argument('--pos_weight', type=float, default=10.0,
                      help='Positive class weight in loss function (default: 10.0)')
    parser.add_argument('--dice_weight', type=float, default=1.0,
                      help='Weight for Dice loss term (default: 1.0)')
    parser.add_argument('--bce_weight', type=float, default=1.0,
                      help='Weight for BCE loss term (default: 1.0)')
    
    # Optimizer settings
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                      help='Initial learning rate (default: 5e-5)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                      help='Weight decay for optimizer (default: 1e-5)')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                      help='Gradient clipping norm (default: 5.0)')
    
    # Runtime settings
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loading workers (default: 4)')
    parser.add_argument('--resume', action='store_true',
                      help='Resume training from last checkpoint')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug output and visualizations')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility (default: 42)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print("\nTraining Configuration:")
    print("-----------------------")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Create config
    config = SegmentationConfig()
    config.data_dir = Path(args.data_dir)
    config.output_dir = Path(args.output_dir)
    config.checkpoint_dir = config.output_dir / 'checkpoints'
    config.log_dir = config.output_dir / 'logs'
    config.num_epochs = args.epochs
    config.resume_training = args.resume
    config.debug = args.debug
    
    # Create output directories
    config.output_dir.mkdir(exist_ok=True, parents=True)
    config.checkpoint_dir.mkdir(exist_ok=True)
    config.log_dir.mkdir(exist_ok=True)
    
    # Create trainer
    trainer = PatchSegmentationTrainer(config)
    
    # Update patch and batch settings
    patch_dims = [int(x) for x in args.patch_size.split(',')]
    trainer.patch_size = tuple(patch_dims)
    trainer.batch_size = args.batch_size
    trainer.num_workers = args.num_workers
    trainer.max_grad_norm = args.grad_clip
    
    # Update dataset settings
    trainer.tumor_only_prob = args.tumor_prob
    
    # Update optimizer settings
    trainer.optimizer.param_groups[0].update({
        'lr': args.learning_rate,
        'weight_decay': args.weight_decay
    })
    
    # Update loss settings
    trainer.criterion.bce.pos_weight = torch.tensor([args.pos_weight]).to(trainer.device)
    trainer.criterion.dice_weight = args.dice_weight
    trainer.criterion.bce_weight = args.bce_weight
    
    # Print configuration
    print("\nModel Configuration:")
    print(f"Patch size: {trainer.patch_size}")
    print(f"Batch size: {trainer.batch_size}")
    print(f"Number of workers: {trainer.num_workers}")
    print(f"Gradient clipping: {trainer.max_grad_norm}")
    print(f"Tumor sampling probability: {trainer.tumor_only_prob}")
    
    print("\nLoss Configuration:")
    print(f"Positive class weight: {args.pos_weight}")
    print(f"Dice loss weight: {args.dice_weight}")
    print(f"BCE loss weight: {args.bce_weight}")
    
    print("\nOptimizer Configuration:")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Weight decay: {args.weight_decay}")
    
    if args.debug:
        print("\nDebug mode enabled - will show additional output and visualizations")
    
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