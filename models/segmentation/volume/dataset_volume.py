import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Tuple, Optional, List, Dict
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)

class KiTS23VolumeDataset(Dataset):
    """Dataset for full-volume training with kidney masking support"""
    
    def __init__(self,
                data_dir: str,
                use_kidney_mask: bool = True,
                sliding_window_size: Optional[Tuple[int, int, int]] = None,
                inference_overlap: float = 0.5,
                debug: bool = False):
        """
        Args:
            data_dir: Path to preprocessed .pt files
            use_kidney_mask: Whether to return 2-channel input (CT + kidney)
            sliding_window_size: Size of sliding window crops (D,H,W) or None for full volume
            inference_overlap: Overlap ratio for sliding window inference
            debug: Enable debug logging
        """
        self.data_dir = Path(data_dir)
        self.use_kidney_mask = use_kidney_mask
        self.sliding_window_size = sliding_window_size
        self.inference_overlap = inference_overlap
        self.debug = debug
        
        # Find all preprocessed volumes
        self.volume_paths = sorted(list(self.data_dir.glob("case_*.pt")))
        if not self.volume_paths:
            raise RuntimeError(f"No preprocessed volumes found in {self.data_dir}")
        
        # Track dataset statistics
        self.volumes_with_tumor = 0
        self.total_tumor_voxels = 0
        
        # Initialize if using sliding windows
        if sliding_window_size is not None:
            self.stride = [int(s * (1 - inference_overlap)) for s in sliding_window_size]
            
        print(f"Found {len(self.volume_paths)} volumes in {data_dir}")
        print(f"Input channels: {2 if use_kidney_mask else 1} (CT{' + kidney mask' if use_kidney_mask else ''})")
        if sliding_window_size:
            print(f"Using sliding windows: size={sliding_window_size}, stride={self.stride}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a volume or sliding window crop with optional kidney mask"""
        # Load preprocessed data
        data = torch.load(self.volume_paths[idx])
        image = data['image']  # [1,D,H,W]
        tumor = data['tumor']  # [1,D,H,W]
        
        # Create kidney mask (kidney + tumor + cyst)
        if self.use_kidney_mask:
            kidney = data['kidney']  # [1,D,H,W]
            cyst = data.get('cyst', torch.zeros_like(kidney))  # [1,D,H,W] 
            kidney_mask = (kidney > 0) | (tumor > 0) | (cyst > 0)  # Include kidney, tumor and cysts
            input_tensor = torch.cat([image, kidney_mask.float()], dim=0)  # [2,D,H,W]
        else:
            input_tensor = image  # [1,D,H,W]
            
        if self.sliding_window_size is None:
            # Return full volume
            return {
                'input': input_tensor,
                'target': tumor,
                'case_id': data['case_id'],
                'shape': image.shape[1:],
                'spacing': data.get('spacing', None)
            }
        else:
            # Get sliding window crop
            D, H, W = self.sliding_window_size
            d = np.random.randint(0, image.shape[1] - D + 1)
            h = np.random.randint(0, image.shape[2] - H + 1)
            w = np.random.randint(0, image.shape[3] - W + 1)
            
            input_crop = input_tensor[:, d:d+D, h:h+H, w:w+W]
            target_crop = tumor[:, d:d+D, h:h+H, w:w+W]
            
            return {
                'input': input_crop,
                'target': target_crop,
                'case_id': data['case_id'],
                'crop_coords': (d,h,w),
                'orig_shape': image.shape[1:]
            }

    def __len__(self):
        return len(self.volume_paths)
        
    def get_train_val_splits(self, 
                          val_ratio: float = 0.2,
                          seed: Optional[int] = None) -> Tuple['KiTS23VolumeDataset', 'KiTS23VolumeDataset']:
        """Split dataset into training and validation sets"""
        if seed is not None:
            np.random.seed(seed)
            
        indices = list(range(len(self)))
        np.random.shuffle(indices)
        
        split = int(np.floor(val_ratio * len(self)))
        train_indices = indices[split:]
        val_indices = indices[:split]
        
        # Create train/val datasets
        train_dataset = KiTS23VolumeDataset(
            data_dir=self.data_dir,
            use_kidney_mask=self.use_kidney_mask,
            sliding_window_size=self.sliding_window_size,
            inference_overlap=self.inference_overlap,
            debug=self.debug
        )
        train_dataset.volume_paths = [self.volume_paths[i] for i in train_indices]
        
        val_dataset = KiTS23VolumeDataset(
            data_dir=self.data_dir,
            use_kidney_mask=self.use_kidney_mask,
            sliding_window_size=self.sliding_window_size,
            inference_overlap=self.inference_overlap,
            debug=self.debug
        )
        val_dataset.volume_paths = [self.volume_paths[i] for i in val_indices]
        
        print(f"\nSplit dataset into:")
        print(f"Train: {len(train_dataset)} volumes")
        print(f"Val: {len(val_dataset)} volumes")
        
        return train_dataset, val_dataset