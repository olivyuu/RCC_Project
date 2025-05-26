import argparse
from pathlib import Path
import torch
import multiprocessing as mp
from config import nnUNetConfig
from trainer_volume import nnUNetVolumeTrainer
from tools.create_experiment import create_experiment_structure

def set_seed(seed: int, deterministic: bool = False, benchmark: bool = True):
    """Set seeds for reproducibility."""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    print(f"Set seed to {seed}")
    print(f"cuDNN Deterministic: {deterministic}, Benchmark: {benchmark}")

def main():
    parser = argparse.ArgumentParser(description='Train nnUNet on full volumes')
    parser.add_argument('--data_dir', type=str, default='/workspace/kits23/dataset',
                      help='Path to the dataset directory')
    parser.add_argument('--patch_weights', type=str, 
                      help='Path to the best patch-trained model weights')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size for full volume training')
    parser.add_argument('--accumulation_steps', type=int, default=4,
                      help='Number of gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='Initial learning rate')
    parser.add_argument('--freeze', action='store_true',
                      help='Freeze encoder layers for first few epochs')
    parser.add_argument('--freeze_epochs', type=int, default=2,
                      help='Number of epochs to keep layers frozen')
    parser.add_argument('--experiment_name', type=str, default='nnUNet_volume',
                      help='Name for the experiment')
    parser.add_argument('--resume', action='store_true',
                      help='Resume training from checkpoint')
    parser.add_argument('--preprocess', type=str, choices=['true', 'false'],
                      help='Whether to preprocess data or use existing preprocessed files')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode with additional logging')
    parser.add_argument('--memory_check', action='store_true',
                      help='Enable memory usage tracking')
    args = parser.parse_args()
    
    # Initialize config with volume training settings
    config = nnUNetConfig()
    config.training_mode = "full_volume"
    config.data_dir = args.data_dir
    config.preprocessed_dir = "preprocessed_volumes"  # Set volume preprocessing directory
    config.vol_batch_size = args.batch_size
    config.vol_gradient_accumulation_steps = args.accumulation_steps
    config.num_epochs = args.epochs
    config.initial_lr = args.lr
    config.experiment_name = args.experiment_name
    config.resume_training = args.resume
    config.preprocess = True if args.preprocess is None else args.preprocess.lower() == 'true'
    
    # Create experiment directory structure
    create_experiment_structure(args.experiment_name)
    
    # Update checkpoint directory to use experiment-specific path
    config.checkpoint_dir = Path(f"experiments/{args.experiment_name}/checkpoints")
    
    # Transfer learning settings
    if args.patch_weights:
        config.transfer_learning = True
        config.patch_weights_path = args.patch_weights
        config.freeze_layers = args.freeze
        config.freeze_epochs = args.freeze_epochs
    
    # Set multiprocessing start method
    mp.set_start_method('spawn')
    
    # Set random seeds
    set_seed(config.seed, config.deterministic, config.benchmark_cudnn)
    
    # Initialize debug logger if requested
    debug_logger = None
    if args.debug or args.memory_check:
        from tools.debug_utils import DebugLogger
        debug_logger = DebugLogger(
            experiment_dir=f"experiments/{config.experiment_name}",
            debug=args.debug
        )
        debug_logger.log_memory("Initial state")
    
    # Create trainer and start training
    trainer = nnUNetVolumeTrainer(config, debug_logger=debug_logger)
    trainer.train(config.data_dir)

if __name__ == '__main__':
    main()