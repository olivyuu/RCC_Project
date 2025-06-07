import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional
import numpy as np

def get_sliding_windows(
    shape: Tuple[int, int, int],
    window_size: Tuple[int, int, int],
    stride: Optional[Tuple[int, int, int]] = None,
    min_overlap: float = 0.5
) -> List[Tuple[slice, slice, slice]]:
    """Generate sliding window coordinates with minimum overlap
    
    Args:
        shape: Full volume shape (D,H,W)
        window_size: Size of each window (d,h,w)
        stride: Optional stride override, computed from min_overlap if None
        min_overlap: Minimum overlap between windows if stride not specified
        
    Returns:
        List of (depth_slice, height_slice, width_slice) for each window
    """
    if stride is None:
        stride = [int(s * (1 - min_overlap)) for s in window_size]
        
    windows = []
    D, H, W = shape
    d, h, w = window_size
    zs, ys, xs = stride
    
    for z in range(0, D - d + 1, zs):
        for y in range(0, H - h + 1, ys):
            for x in range(0, W - w + 1, xs):
                windows.append((
                    slice(z, z + d),
                    slice(y, y + h), 
                    slice(x, x + w)
                ))
                
    return windows

def stitch_predictions(
    windows: List[torch.Tensor],
    window_coords: List[Tuple[slice, slice, slice]],
    output_shape: Tuple[int, int, int, int],
    overlap_mode: str = 'mean'
) -> torch.Tensor:
    """Stitch together overlapping window predictions
    
    Args:
        windows: List of predicted window tensors [B,C,d,h,w]
        window_coords: List of window coordinates (from get_sliding_windows)
        output_shape: Target output shape [B,C,D,H,W]
        overlap_mode: How to handle overlapping regions ('mean' or 'max')
        
    Returns:
        Stitched volume of shape output_shape
    """
    device = windows[0].device
    dtype = windows[0].dtype
    
    # Initialize output volume and count map
    output = torch.zeros(output_shape, dtype=dtype, device=device)
    counts = torch.zeros(output_shape, dtype=dtype, device=device)
    
    # Add each window prediction
    for pred, coords in zip(windows, window_coords):
        z_slice, y_slice, x_slice = coords
        output[..., z_slice, y_slice, x_slice] += pred
        counts[..., z_slice, y_slice, x_slice] += 1
        
    # Average overlapping regions
    valid_mask = counts > 0
    if overlap_mode == 'mean':
        output[valid_mask] /= counts[valid_mask]
    else:  # max mode
        output = torch.where(valid_mask, output / counts, output)
        
    return output

def get_required_padding(
    shape: Tuple[int, ...],
    window_size: Tuple[int, ...],
    stride: Tuple[int, ...]
) -> List[Tuple[int, int]]:
    """Calculate padding needed for seamless sliding windows
    
    Args:
        shape: Input tensor shape (excluding batch/channel dims)
        window_size: Size of sliding windows
        stride: Stride between windows
        
    Returns:
        List of (pad_left, pad_right) for each dimension
    """
    padding = []
    for s, w, st in zip(shape, window_size, stride):
        # Calculate total padding needed
        n_windows = max(1, (s - w) // st + 1)
        total_coverage = (n_windows - 1) * st + w
        pad_needed = max(0, total_coverage - s)
        
        # Split padding evenly
        pad_left = pad_needed // 2
        pad_right = pad_needed - pad_left
        padding.append((pad_left, pad_right))
        
    return padding[::-1]  # Reverse for torch.nn.functional.pad