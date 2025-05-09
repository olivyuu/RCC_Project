from dataclasses import dataclass
from typing import Tuple
from pathlib import Path

@dataclass
class nnUNetConfig:
    # Data parameters
    data_dir: str = "/workspace/kits23/data"
    preprocessed_dir: str = "preprocessed_patches"
    # Model parameters
    in_channels: int = 1
    out_channels: int = 2  # Binary segmentation (background, tumor)
    features: Tuple[int, ...] = (32, 64, 128, 256, 320)
    
    # Training parameters
    batch_size: int = 4  # Optimized for RTX 3070
    validation_split: float = 0.2
    
    # Patch parameters - Optimized for KiTS23 dimensions
    patch_size: Tuple[int, int, int] = (32, 96, 96)  # D, H, W - More memory efficient while maintaining detail
    patch_stride: Tuple[int, int, int] = (16, 48, 48)  # 50% overlap
    min_patches_per_image: int = 10
    max_patches_per_chunk: int = 100
    
    # Memory management - Optimized for RTX 3070
    chunk_size: int = 8
    max_chunk_memory: int = 512  # MB, adjust based on GPU memory
    dtype: str = 'float32'
    num_workers: int = 4  # Increased for better data loading
    pin_memory: bool = True
    prefetch_factor: int = 4  # Increased for smoother data pipeline
    
    # Training loop parameters
    num_epochs: int = 15
    early_stopping_patience: int = 75  # Increased for longer training
    save_frequency: int = 5  # More frequent checkpoints
    
    # Optimizer parameters
    initial_lr: float = 2e-4  # Slightly higher initial learning rate
    weight_decay: float = 2e-5  # Slightly increased regularization
    lr_schedule_patience: int = 15  # Adjusted for longer training
    lr_reduce_factor: float = 0.7  # Less aggressive LR reduction
    
    # Checkpoint parameters
    resume_training: bool = True
    checkpoint_dir: Path = Path("checkpoints")
    checkpoint_file: str = "latest.pth"
    save_latest: bool = True
    
    # Augmentation parameters
    rotation_angle: Tuple[int, int] = (-30, 30)  # Increased rotation range
    scale_range: Tuple[float, float] = (0.8, 1.2)  # Increased scale range
    elastic_deformation_alpha: Tuple[int, int] = (0, 600)  # Increased deformation
    elastic_deformation_sigma: Tuple[int, int] = (10, 15)  # Adjusted sigma range
    
    # Cross-validation parameters
    fold: int = 0
    num_folds: int = 5
    
    # Logging parameters
    log_dir: str = "runs"
    experiment_name: str = "nnUNet_v1"
    
    # System parameters
    seed: int = 42  # Seed for reproducibility
    use_amp: bool = True
    benchmark_cudnn: bool = True
    deterministic: bool = False
    
    # Size thresholds for preprocessing
    max_image_size_mb: int = 1024  # Skip images larger than 1GB
    max_image_dimensions: Tuple[int, int, int] = (512, 512, 512)  # Max dimensions based on research showing no performance gain beyond 512x512
    downsample_large_images: bool = True  # Downsample images larger than max dimensions
    downsample_factor: int = 2  # Factor by which to downsample large images (to reach target size)
max_estimated_memory_mb: int = 1536 # Skip images estimated > 1.5GB uncompressed