import argparse
from pathlib import Path
import logging
import sys

from models.segmentation.volume.volume_trainer import VolumeSegmentationTrainer
from models.segmentation.volume.volume_config import VolumeSegmentationConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Train segmentation model on full volumes")
    
    # Dataset parameters
    parser.add_argument("--data_dir", type=str, required=True,
                      help="Directory containing preprocessed volumes")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to save model checkpoints")
    
    # Model parameters  
    parser.add_argument("--features", type=int, default=32,
                      help="Number of base features in UNet")
    parser.add_argument("--batch_size", type=int, default=1,
                      help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50,
                      help="Number of epochs to train")
                      
    # Training parameters
    parser.add_argument("--phase", type=int, choices=[3,4], required=True,
                      help="Training phase (3=with kidney mask, 4=without)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                      help="Initial learning rate")
    parser.add_argument("--grad_clip", type=float, default=5.0,
                      help="Gradient clipping value")
                      
    # Sliding window parameters
    parser.add_argument("--window_size", type=str, default="64,128,128",
                      help="Sliding window size (D,H,W)")
    parser.add_argument("--window_overlap", type=float, default=0.5,
                      help="Overlap ratio between windows")
    parser.add_argument("--min_tumor_size", type=int, default=1000,
                      help="Minimum tumor size after postprocessing")
    
    # Warmup parameters
    parser.add_argument("--warmup_epochs", type=int, default=0,
                      help="Number of epochs to train on tumor-rich windows (0 to disable)")
    parser.add_argument("--warmup_window_size", type=str, default="64,128,128",
                      help="Size of tumor-rich windows during warmup (D,H,W)")
    parser.add_argument("--warmup_window_stride", type=str, default="32,64,64",
                      help="Stride between tumor-rich windows (D,H,W)")
    parser.add_argument("--warmup_min_tumor_ratio", type=float, default=0.001,
                      help="Minimum tumor/total voxels ratio to keep window")
    parser.add_argument("--warmup_top_percent", type=float, default=20.0,
                      help="Keep only top K% most tumor-rich windows")
    
    # Checkpoint parameters
    parser.add_argument("--checkpoint_path", type=str,
                      help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug logging and visualization")
    
    args = parser.parse_args()
    return args

def setup_logging(output_dir: Path):
    """Configure logging to file and console"""
    log_file = output_dir / "training.log"
    
    # Configure file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir)
    
    # Parse window sizes
    window_size = tuple(map(int, args.window_size.split(",")))
    warmup_window_size = tuple(map(int, args.warmup_window_size.split(",")))
    warmup_window_stride = tuple(map(int, args.warmup_window_stride.split(",")))
    
    # Create configuration
    config = VolumeSegmentationConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        training_phase=args.phase,
        features=args.features,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        sliding_window_size=window_size,
        inference_overlap=args.window_overlap,
        postproc_min_size=args.min_tumor_size,
        warmup_epochs=args.warmup_epochs,
        warmup_window_size=warmup_window_size,
        warmup_window_stride=warmup_window_stride,
        warmup_min_tumor_ratio=args.warmup_min_tumor_ratio,
        warmup_top_percent=args.warmup_top_percent,
        learning_rate=args.learning_rate,
        grad_clip=args.grad_clip,
        checkpoint_path=args.checkpoint_path,
        debug=args.debug
    )
    
    # Save configuration
    config.save(output_dir / "volume_config.json")
    
    # Create trainer
    trainer = VolumeSegmentationTrainer(config)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()