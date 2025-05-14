"""
Debugging utilities for volume training and memory tracking.
"""
import torch
import psutil
import os
import numpy as np
from pathlib import Path
import logging
from typing import Union, Tuple, List
from contextlib import contextmanager
import time
import gc

class DebugLogger:
    def __init__(self, experiment_dir: Union[str, Path], debug: bool = False):
        self.debug = debug
        log_dir = Path(experiment_dir) / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger('debug_logger')
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_dir / 'debug.log')
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def log_memory(self, tag: str = ""):
        """Log current memory usage."""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**2
            gpu_cached = torch.cuda.memory_reserved() / 1024**2
            self.logger.debug(f"{tag} GPU Memory: {gpu_memory:.1f}MB (Cached: {gpu_cached:.1f}MB)")
        
        process = psutil.Process(os.getpid())
        ram_usage = process.memory_info().rss / 1024**2
        self.logger.debug(f"{tag} RAM Usage: {ram_usage:.1f}MB")
    
    def log_shapes(self, tag: str, **tensors):
        """Log shapes of multiple tensors."""
        shapes = {name: tuple(t.shape) for name, t in tensors.items()}
        self.logger.debug(f"{tag} Shapes: {shapes}")
    
    def log_stats(self, tag: str, **tensors):
        """Log basic statistics of tensors."""
        for name, t in tensors.items():
            if torch.is_tensor(t):
                stats = {
                    'min': float(t.min()),
                    'max': float(t.max()),
                    'mean': float(t.mean()),
                    'std': float(t.std()),
                    'has_nan': torch.isnan(t).any().item(),
                    'has_inf': torch.isinf(t).any().item()
                }
                self.logger.debug(f"{tag} {name} Stats: {stats}")
    
    @contextmanager
    def track_time(self, tag: str):
        """Context manager to track execution time."""
        start = time.time()
        yield
        duration = time.time() - start
        self.logger.debug(f"{tag} Time: {duration:.2f}s")

def validate_volume_shapes(volume: torch.Tensor, expected_dims: int = 5):
    """Validate volume tensor shapes."""
    if not isinstance(volume, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(volume)}")
    
    if len(volume.shape) != expected_dims:
        raise ValueError(
            f"Expected {expected_dims} dimensions (B,C,D,H,W), "
            f"got shape {volume.shape}"
        )
    
    if volume.shape[1] not in [1, 2]:  # Check channels
        raise ValueError(f"Expected 1 or 2 channels, got {volume.shape[1]}")

def check_memory_usage(shapes: List[Tuple[int, ...]], dtype=torch.float32) -> float:
    """Estimate memory usage for given tensor shapes."""
    total_elements = sum(np.prod(shape) for shape in shapes)
    bytes_per_element = torch.ones(1, dtype=dtype).element_size()
    memory_mb = total_elements * bytes_per_element / 1024**2
    return memory_mb

@contextmanager
def gpu_memory_check(threshold_mb: float = 1000):
    """Context manager to check GPU memory changes."""
    if not torch.cuda.is_available():
        yield
        return
    
    start_mem = torch.cuda.memory_allocated() / 1024**2
    try:
        yield
    finally:
        end_mem = torch.cuda.memory_allocated() / 1024**2
        change = end_mem - start_mem
        if change > threshold_mb:
            logging.warning(
                f"Large GPU memory increase: {change:.1f}MB "
                f"(from {start_mem:.1f}MB to {end_mem:.1f}MB)"
            )

def debug_data_sample(sample: Tuple[torch.Tensor, torch.Tensor], save_dir: Union[str, Path]):
    """Save debug information about a data sample."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    image, mask = sample
    
    info = {
        'image_shape': tuple(image.shape),
        'mask_shape': tuple(mask.shape),
        'image_stats': {
            'min': float(image.min()),
            'max': float(image.max()),
            'mean': float(image.mean()),
            'std': float(image.std()),
            'has_nan': bool(torch.isnan(image).any()),
            'has_inf': bool(torch.isinf(image).any())
        },
        'mask_stats': {
            'unique_values': sorted([int(x) for x in torch.unique(mask).tolist()]),
            'has_nan': bool(torch.isnan(mask).any()),
            'has_inf': bool(torch.isinf(mask).any())
        }
    }
    
    import json
    with open(save_dir / 'sample_debug.json', 'w') as f:
        json.dump(info, f, indent=2)

def force_cuda_sync():
    """Force CUDA synchronization and clear cache."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()

# Example usage:
"""
# Initialize debug logger
debug = DebugLogger("experiments/volume_ft_001", debug=True)

# Track memory and time
with debug.track_time("forward_pass"):
    with gpu_memory_check():
        output = model(input)
        debug.log_memory("After forward pass")
        debug.log_shapes("Output", output=output, target=target)
        debug.log_stats("Output", output=output)

# Validate volume shapes
validate_volume_shapes(volume)

# Check memory requirements
memory_mb = check_memory_usage([
    (1, 1, 128, 256, 256),  # Input
    (1, 2, 128, 256, 256)   # Output
])
print(f"Estimated memory usage: {memory_mb:.1f}MB")
"""