import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, List, Dict
import numpy as np
import logging
from pathlib import Path
from .dataset_volume import KiTS23VolumeDataset
from .utils.sliding_window import get_sliding_windows

logger = logging.getLogger(__name__)

class TumorRichWindowDataset(KiTS23VolumeDataset):
    """Dataset that samples tumor-rich windows from volumes"""
    
    def __init__(self,
                data_dir: str,
                window_size: Tuple[int, int, int] = (64, 128, 128),
                stride: Optional[Tuple[int, int, int]] = None,
                min_tumor_ratio: float = 0.001,  # Minimum tumor content to keep window
                top_percent: Optional[float] = None,  # Only keep top K% most tumor-rich windows
                use_kidney_mask: bool = False,
                training: bool = True,
                debug: bool = False):
        """
        Args:
            data_dir: Path to preprocessed volumes
            window_size: Size of sliding windows (D,H,W)
            stride: Optional stride override (computed from 50% overlap if None)
            min_tumor_ratio: Minimum tumor/total voxels ratio to keep window
            top_percent: If set, only keep top K% most tumor-rich windows
            use_kidney_mask: Whether to use kidney mask channel
            training: Whether this is training set
            debug: Enable debug output
        """
        super().__init__(data_dir, use_kidney_mask, training, debug)
        
        self.window_size = window_size
        self.stride = stride if stride else tuple(s//2 for s in window_size)
        self.min_tumor_ratio = min_tumor_ratio
        self.top_percent = top_percent
        
        # Store (volume_idx, window_coords) pairs for valid windows
        self.window_indices = []
        self.tumor_ratios = []
        
        logger.info("Computing tumor-rich windows...")
        self._precompute_windows()
        
    def _precompute_windows(self):
        """Find all valid windows and their tumor ratios"""
        total_windows = 0
        valid_windows = 0
        
        for vol_idx in range(len(self)):
            # Load volume data
            data = torch.load(self.volume_paths[vol_idx])
            tumor = data['tumor'][0]  # Remove channel dim
            
            # Get all possible windows
            windows = get_sliding_windows(
                shape=tumor.shape,
                window_size=self.window_size,
                stride=self.stride
            )
            
            total_windows += len(windows)
            
            # Calculate tumor ratio for each window
            for window in windows:
                tumor_patch = tumor[window[0], window[1], window[2]]
                ratio = (tumor_patch > 0).float().mean().item()
                
                if ratio >= self.min_tumor_ratio:
                    self.window_indices.append((vol_idx, window))
                    self.tumor_ratios.append(ratio)
                    valid_windows += 1
        
        # Sort windows by tumor ratio (highest first)
        sorted_indices = np.argsort(self.tumor_ratios)[::-1]
        self.window_indices = [self.window_indices[i] for i in sorted_indices]
        self.tumor_ratios = [self.tumor_ratios[i] for i in sorted_indices]
        
        # Optionally keep only top K% windows
        if self.top_percent is not None:
            n_keep = int(len(self.window_indices) * self.top_percent / 100)
            self.window_indices = self.window_indices[:n_keep]
            self.tumor_ratios = self.tumor_ratios[:n_keep]
        
        logger.info(f"Found {valid_windows}/{total_windows} valid windows")
        logger.info(f"Keeping {len(self.window_indices)} windows")
        if len(self.tumor_ratios) > 0:
            logger.info(f"Tumor ratios: min={min(self.tumor_ratios):.4f}, "
                     f"max={max(self.tumor_ratios):.4f}, "
                     f"mean={np.mean(self.tumor_ratios):.4f}")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a tumor-rich window"""
        vol_idx, window = self.window_indices[idx]
        
        # Load volume data
        data = torch.load(self.volume_paths[vol_idx])
        image = data['image']         # [1,D,H,W]
        tumor = data['tumor']         # [1,D,H,W]
        
        # Extract window
        image_patch = image[:, window[0], window[1], window[2]]
        tumor_patch = tumor[:, window[0], window[1], window[2]]
        
        if self.use_kidney_mask:
            kidney = data['kidney']  # [1,D,H,W]
            kidney_patch = kidney[:, window[0], window[1], window[2]]
            input_tensor = torch.cat([image_patch, kidney_patch], dim=0)
        else:
            input_tensor = image_patch
        
        return {
            'input': input_tensor,
            'target': tumor_patch,
            'case_id': f"{data['case_id']}_w{idx}",
            'tumor_ratio': self.tumor_ratios[idx]
        }
    
    def __len__(self):
        return len(self.window_indices)