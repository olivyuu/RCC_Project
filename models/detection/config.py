from dataclasses import dataclass
from typing import Tuple
from pathlib import Path

@dataclass
class DetectionConfig:
    # Data parameters
    data_dir: str = "/workspace/kits23/dataset"
    preprocessed_dir: str = "preprocessed_volumes"
    
    # Model parameters
    in_channels: int = 1
    num_classes: int = 2  # Binary classification (no tumor, tumor)
    features: Tuple[int, ...] = (32, 64, 128, 256, 320)
    
    # Training parameters
    batch_size: int = 4
    validation_split: float = 0.2
    
    # Volume parameters
    vol_max_dim: Tuple[int, int, int] = (128, 256, 256)  # Max dimensions for volume processing
    
    # Memory management
    dtype: str = 'float32'
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 4
    
    # Training loop parameters
    num_epochs: int = 50
    early_stopping_patience: int = 20
    save_frequency: int = 5
    
    # Optimizer parameters
    initial_lr: float = 1e-4
    weight_decay: float = 1e-4
    lr_schedule_patience: int = 10
    lr_reduce_factor: float = 0.5
    
    # Checkpoint parameters
    resume_training: bool = True
    checkpoint_dir: Path = Path("checkpoints/detection")
    checkpoint_file: str = "latest.pth"
    save_latest: bool = True
    
    # Augmentation parameters - simplified for detection
    rotation_angle: Tuple[int, int] = (-15, 15)
    scale_range: Tuple[float, float] = (0.9, 1.1)
    
    # Cross-validation parameters
    fold: int = 0
    num_folds: int = 5
    
    # Logging parameters
    log_dir: str = "runs/detection"
    experiment_name: str = "DetNet_v1"
    
    # System parameters
    seed: int = 42
    use_amp: bool = True
    benchmark_cudnn: bool = True
    deterministic: bool = False
    debug: bool = False
    memory_check: bool = False
    preprocess: bool = True
    
    # Size thresholds for preprocessing
    max_image_size_mb: int = 1024
    max_image_dimensions: Tuple[int, int, int] = (512, 512, 512)
    downsample_large_images: bool = True
    downsample_factor: int = 2
    max_estimated_memory_mb: int = 1536
    
    # Loss parameters
    class_weights: Tuple[float, float] = (1.0, 2.0)  # Weight tumor class higher
    focal_loss_gamma: float = 2.0  # For handling class imbalance
    
    # Inference parameters
    confidence_threshold: float = 0.5