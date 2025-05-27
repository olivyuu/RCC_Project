from dataclasses import dataclass
from typing import Tuple
from pathlib import Path

@dataclass
class SegmentationConfig:
    # Training mode
    training_mode: str = "patch"  # Options: "patch", "full_volume"
    
    # Data parameters
    data_dir: str = "/workspace/kits23/dataset"
    preprocessed_dir: str = "preprocessed_patches"
    preprocessed_volumes_dir: str = "preprocessed_volumes"  # For full volume training
    
    # Model parameters
    in_channels: int = 1
    out_channels: int = 2  # Binary segmentation (background, tumor)
    features: Tuple[int, ...] = (32, 64, 128, 256, 320)
    
    # Training parameters
    batch_size: int = 4  # Will be adjusted based on training_mode
    validation_split: float = 0.2
    
    # Full volume parameters
    vol_batch_size: int = 1  # Smaller batch size for full volumes
    vol_gradient_accumulation_steps: int = 4  # Accumulate gradients for effective batch size
    vol_max_dim: Tuple[int, int, int] = (128, 256, 256)  # Max dimensions for full volume training
    
    # Patch parameters - Optimized for KiTS23 dimensions
    patch_size: Tuple[int, int, int] = (32, 96, 96)  # D, H, W - More memory efficient while maintaining detail
    patch_stride: Tuple[int, int, int] = (16, 48, 48)  # 50% overlap
    min_patches_per_image: int = 10
    max_patches_per_chunk: int = 100
    
    # Memory management - Optimized for RTX 3070
    chunk_size: int = 8
    max_chunk_memory: int = 512  # MB, adjust based on GPU memory
    dtype: str = 'float32'
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 4
    
    # Training loop parameters
    num_epochs: int = 15
    early_stopping_patience: int = 75
    save_frequency: int = 5
    
    # Optimizer parameters
    initial_lr: float = 2e-4
    weight_decay: float = 2e-5
    lr_schedule_patience: int = 15
    lr_reduce_factor: float = 0.7
    
    # Checkpoint parameters
    resume_training: bool = True
    checkpoint_dir: Path = Path("checkpoints/segmentation")
    checkpoint_file: str = "latest.pth"
    save_latest: bool = True
    
    # Augmentation parameters
    rotation_angle: Tuple[int, int] = (-30, 30)
    scale_range: Tuple[float, float] = (0.8, 1.2)
    elastic_deformation_alpha: Tuple[int, int] = (0, 600)
    elastic_deformation_sigma: Tuple[int, int] = (10, 15)
    
    # Cross-validation parameters
    fold: int = 0
    num_folds: int = 5
    
    # Logging parameters
    log_dir: str = "runs/segmentation"
    experiment_name: str = "SegNet_v1"
    
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
    
    # Transfer learning parameters
    transfer_learning: bool = False
    patch_weights_path: str = ""
    freeze_layers: bool = False
    freeze_epochs: int = 2