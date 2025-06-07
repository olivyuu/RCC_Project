import json
from pathlib import Path
from typing import Tuple, Optional, Any, Dict

class VolumeSegmentationConfig:
    """Configuration for full-volume segmentation training"""
    
    def __init__(self,
                data_dir: str,
                output_dir: str,
                training_phase: int = 3,
                features: int = 32,
                batch_size: int = 1,
                num_epochs: int = 50,
                # Volume-specific parameters
                sliding_window_size: Optional[Tuple[int, int, int]] = (64, 128, 128),
                inference_overlap: float = 0.5,
                postproc_min_size: int = 1000,
                # Training parameters
                learning_rate: float = 1e-4,
                grad_clip: float = 5.0,
                checkpoint_path: Optional[str] = None,
                debug: bool = False):
        """Initialize volume training configuration
        
        Args:
            data_dir: Directory containing preprocessed volumes
            output_dir: Directory for training outputs
            training_phase: Training phase (3=with kidney mask, 4=without)
            features: Number of base features in UNet
            batch_size: Batch size for training (usually 1 for full volumes)
            num_epochs: Number of epochs to train
            sliding_window_size: Size of sliding window for inference
            inference_overlap: Overlap between sliding windows
            postproc_min_size: Minimum tumor size after postprocessing
            learning_rate: Initial learning rate
            grad_clip: Gradient clipping value
            checkpoint_path: Path to checkpoint to resume from
            debug: Enable debug output
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Phase configuration
        self.training_phase = training_phase
        self.use_kidney_mask = (training_phase == 3)
        self.in_channels = 2 if self.use_kidney_mask else 1
        
        # Model parameters
        self.features = features
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        # Volume-specific parameters
        self.sliding_window_size = sliding_window_size
        self.inference_overlap = inference_overlap
        self.postproc_min_size = postproc_min_size
        
        # Training parameters
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.checkpoint_path = checkpoint_path
        
        self.debug = debug
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization"""
        return {
            'data_dir': str(self.data_dir),
            'output_dir': str(self.output_dir),
            'training_phase': self.training_phase,
            'use_kidney_mask': self.use_kidney_mask,
            'in_channels': self.in_channels,
            'features': self.features,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'sliding_window_size': self.sliding_window_size,
            'inference_overlap': self.inference_overlap,
            'postproc_min_size': self.postproc_min_size,
            'learning_rate': self.learning_rate,
            'grad_clip': self.grad_clip,
            'checkpoint_path': self.checkpoint_path,
            'debug': self.debug
        }
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'VolumeSegmentationConfig':
        """Create config from dictionary"""
        # Handle paths
        data_dir = config_dict.pop('data_dir')
        output_dir = config_dict.pop('output_dir')
        
        # Remove derived attributes
        config_dict.pop('use_kidney_mask', None)
        config_dict.pop('in_channels', None)
        
        return cls(data_dir=data_dir, output_dir=output_dir, **config_dict)
        
    def save(self, path: Path):
        """Save config to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    @classmethod
    def load(cls, path: Path) -> 'VolumeSegmentationConfig':
        """Load config from JSON file"""
        with open(path) as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
        
    def __str__(self) -> str:
        """String representation with key parameters"""
        phase_name = "Phase 3 (with kidney mask)" if self.use_kidney_mask else "Phase 4 (without kidney mask)"
        return f"""Volume Segmentation Config:
Training Phase: {phase_name}
Input Channels: {self.in_channels}
Features: {self.features}
Batch Size: {self.batch_size}
Sliding Window: {self.sliding_window_size}
Window Overlap: {self.inference_overlap:.1%}
Learning Rate: {self.learning_rate}"""