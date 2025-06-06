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
        self.checkpoint_path: Optional[str] = None
        
        # Data configuration
        self.image_size: Tuple[int, int, int] = (128, 256, 256)  # D, H, W
        self.patch_size: Tuple[int, int, int] = (64, 128, 128)   # For patch training
        
        # Phase control
        self.training_phase: int = 1  # Track which phase we're in
        self.use_kidney_mask: bool = False  # Phase 2: Enable kidney masking
        self.min_kidney_voxels: int = 100  # Minimum kidney voxels for valid patch
        self.kidney_patch_overlap: float = 0.5  # Minimum overlap with kidney
        
        # Model configuration
        self.in_channels: int = 1  # CT only for Phase 1
        self.out_channels: int = 1  # Binary tumor segmentation
        self.features: int = 32     # Base feature channels
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
        self.mixed_precision: bool = True  # Use mixed precision training
        self.deterministic: bool = True   # Make training reproducible
        
        # Hardware settings
        self.gpu_mem_fraction: float = 0.95  # Maximum GPU memory fraction to use

    def update(self, **kwargs):
        """Update config attributes from keyword arguments"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
    
    def __repr__(self):
        """String representation of config"""
        attrs = [
            f"  {key}={value}"
            for key, value in self.__dict__.items()
            if not key.startswith('_')
        ]
        return "SegmentationConfig:\n" + "\n".join(sorted(attrs))
    
    def save(self, path: Union[str, Path]):
        """Save configuration to file"""
        import json
        
        path = Path(path)
        # Convert paths to strings for JSON serialization
        config_dict = {
            k: str(v) if isinstance(v, Path) else v 
            for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]):
        """Load configuration from file"""
        import json
        
        path = Path(path)
        config = cls()
        
        with open(path) as f:
            config_dict = json.load(f)
            
        # Convert string paths back to Path objects
        for key, value in config_dict.items():
            if key.endswith('_dir') or key.endswith('_path'):
                value = Path(value)
            setattr(config, key, value)
            
        return config

    def get_phase_name(self) -> str:
        """Get string name of current training phase"""
        phase_names = {
            1: "Phase 1: Basic Patch Training",
            2: "Phase 2: Kidney-Aware Patch Training",
            3: "Phase 3: Volume Training"
        }
        return phase_names.get(self.training_phase, "Unknown Phase")