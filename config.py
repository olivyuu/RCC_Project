from pathlib import Path
from models.detection.config import DetectionConfig
from models.segmentation.config import SegmentationConfig

# Default configurations
detection_config = DetectionConfig()
segmentation_config = SegmentationConfig()

# Shared system paths and settings that apply to both models
class SharedConfig:
    # Base paths
    PROJECT_ROOT = Path(".")
    DATA_ROOT = Path("/workspace/kits23/dataset")
    CHECKPOINT_ROOT = Path("checkpoints")
    
    # Dataset splits
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # Common preprocessing settings
    NORMALIZE = True
    RESAMPLE_SPACING = (1.0, 1.0, 1.0)  # Target spacing in mm
    HU_MIN = -1024
    HU_MAX = 1024
    
    # Hardware settings
    NUM_WORKERS = 4
    PIN_MEMORY = True
    USE_AMP = True
    
    # Random seed
    SEED = 42

# Create model-specific directories
def create_model_directories():
    """Create necessary directories for both models if they don't exist."""
    for path in [
        detection_config.checkpoint_dir,
        detection_config.log_dir,
        segmentation_config.checkpoint_dir,
        segmentation_config.log_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

# Function to get appropriate config based on task
def get_config(task: str):
    """
    Get the configuration for the specified task.
    
    Args:
        task: Either 'detection' or 'segmentation'
        
    Returns:
        The corresponding configuration object
    """
    if task.lower() == 'detection':
        return detection_config
    elif task.lower() == 'segmentation':
        return segmentation_config
    else:
        raise ValueError(f"Unknown task: {task}. Must be either 'detection' or 'segmentation'")

# Create directories on import
create_model_directories()