from pathlib import Path
from typing import Optional, List, Tuple, Union

class SegmentationConfig:
    """Configuration for segmentation model training"""
    def __init__(self):
        # Data paths
        self.data_dir: Optional[Path] = None
        self.output_dir: Optional[Path] = None
        self.checkpoint_dir: Optional[Path] = None
        self.log_dir: Optional[Path] = None
        
        # Data configuration
        self.image_size: Tuple[int, int, int] = (128, 256, 256)  # D, H, W
        self.patch_size: Tuple[int, int, int] = (64, 128, 128)   # For patch training
        
        # Model configuration
        self.in_channels: int = 1  # CT only for Phase 1
        self.out_channels: int = 1  # Binary tumor segmentation
        self.features: int = 32     # Base feature channels (as scalar)
        self.dropout: float = 0.1
        
        # Training settings
        self.batch_size: int = 2
        self.num_workers: int = 4
        self.learning_rate: float = 5e-5
        self.weight_decay: float = 1e-5
        self.max_grad_norm: float = 5.0
        self.num_epochs: int = 100
        
        # Patch sampling settings
        self.tumor_prob: float = 0.7      # Probability of tumor-centered patches
        self.min_tumor_voxels: int = 100  # Minimum tumor voxels for sampling
        
        # Loss function settings
        self.pos_weight: float = 10.0     # Positive class weight
        self.dice_weight: float = 1.0     # Weight for Dice loss
        self.bce_weight: float = 1.0      # Weight for BCE loss
        
        # Optimizer settings
        self.beta1: float = 0.9           # Adam beta1
        self.beta2: float = 0.999         # Adam beta2
        self.epsilon: float = 1e-8        # Adam epsilon
        
        # Learning rate scheduler
        self.scheduler_factor: float = 0.5     # Multiply LR by this on plateau
        self.scheduler_patience: int = 5       # Epochs to wait before reducing LR
        self.scheduler_threshold: float = 1e-4 # Minimum improvement to reset patience
        self.scheduler_min_lr: float = 1e-6    # Minimum learning rate
        
        # Validation settings
        self.val_interval: int = 1        # Validate every N epochs
        self.val_percent: float = 0.2     # Fraction of data for validation
        
        # Runtime settings
        self.debug: bool = False          # Enable debug output
        self.resume_training: bool = False # Resume from checkpoint
        self.save_best_only: bool = True  # Only save best model
        self.preprocess: bool = False     # Preprocess data on load
        self.mixed_precision: bool = True # Use mixed precision training
        self.deterministic: bool = True   # Make training reproducible
        
        # Hardware settings
        self.gpu_mem_fraction: float = 0.95  # Maximum GPU memory fraction to use
        
        # Phase tracking
        self.current_phase: int = 1  # Track which training phase we're in

    def update(self, **kwargs):
        """Update config attributes from keyword arguments"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
    
    def __repr__(self):
        """String representation of config"""
        attrs = [f"  {key}={value}" for key, value in self.__dict__.items()]
        return "SegmentationConfig:\n" + "\n".join(attrs)