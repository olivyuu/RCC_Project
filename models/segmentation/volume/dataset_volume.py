import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, List, Dict
from pathlib import Path
import logging
import numpy as np
import nibabel as nib

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
            data_dir: Path to case folders containing .nii.gz files
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
        
        # Find all case folders
        self.case_paths = sorted([f for f in self.data_dir.iterdir() if f.is_dir()])
        if not self.case_paths:
            raise RuntimeError(f"No case folders found in {self.data_dir}")
        
        logger.info(f"Found {len(self.case_paths)} cases in {data_dir}")
        logger.info(f"Input channels: {2 if use_kidney_mask else 1} (CT{' + kidney mask' if use_kidney_mask else ''})")
        if sliding_window_size:
            stride = [int(s * (1 - inference_overlap)) for s in sliding_window_size]
            logger.info(f"Using sliding windows: size={sliding_window_size}, stride={stride}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a volume or sliding window crop"""
        case_path = self.case_paths[idx]
        
        # Load CT volume
        img_path = case_path / "imaging.nii.gz"
        img_obj = nib.load(str(img_path))
        ct_data = img_obj.get_fdata()
        
        # Load segmentation
        seg_path = case_path / "segmentation.nii.gz"
        seg_obj = nib.load(str(seg_path))
        seg_data = seg_obj.get_fdata()
        
        # Create masks
        tumor_mask = (seg_data == 2).astype(np.float32)  # Tumor
        
        # Combined kidney mask includes kidney, tumor, and cysts
        kidney_mask = ((seg_data == 1) | (seg_data == 2) | (seg_data == 3)).astype(np.float32)
        
        # Normalize CT globally
        ct_min, ct_max = ct_data.min(), ct_data.max()
        if ct_max > ct_min:
            ct_norm = (ct_data - ct_min) / (ct_max - ct_min)
        else:
            ct_norm = np.zeros_like(ct_data)
            
        # Convert to tensors with channel dimension
        ct_tensor = torch.from_numpy(ct_norm)[None]          # [1,D,H,W]
        tumor_tensor = torch.from_numpy(tumor_mask)[None]    # [1,D,H,W]
        
        if self.use_kidney_mask:
            kidney_tensor = torch.from_numpy(kidney_mask)[None]  # [1,D,H,W]
            input_tensor = torch.cat([ct_tensor, kidney_tensor], dim=0)  # [2,D,H,W]
        else:
            input_tensor = ct_tensor  # [1,D,H,W]
            
        if self.sliding_window_size is None:
            # Return full volume
            return {
                'input': input_tensor,
                'target': tumor_tensor,
                'case_id': case_path.name,
                'shape': ct_data.shape,
                'spacing': img_obj.header.get_zooms()
            }
        else:
            # Get random sliding window crop
            D, H, W = self.sliding_window_size
            d = np.random.randint(0, ct_data.shape[0] - D + 1)
            h = np.random.randint(0, ct_data.shape[1] - H + 1)
            w = np.random.randint(0, ct_data.shape[2] - W + 1)
            
            input_crop = input_tensor[:, d:d+D, h:h+H, w:w+W]
            target_crop = tumor_tensor[:, d:d+D, h:h+H, w:w+W]
            
            return {
                'input': input_crop,
                'target': target_crop,
                'case_id': case_path.name,
                'crop_coords': (d,h,w),
                'orig_shape': ct_data.shape
            }

    def __len__(self):
        return len(self.case_paths)
        
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
            data_dir=str(self.data_dir),
            use_kidney_mask=self.use_kidney_mask,
            sliding_window_size=self.sliding_window_size,
            inference_overlap=self.inference_overlap,
            debug=self.debug
        )
        train_dataset.case_paths = [self.case_paths[i] for i in train_indices]
        
        val_dataset = KiTS23VolumeDataset(
            data_dir=str(self.data_dir),
            use_kidney_mask=self.use_kidney_mask,
            sliding_window_size=self.sliding_window_size,
            inference_overlap=self.inference_overlap,
            debug=self.debug
        )
        val_dataset.case_paths = [self.case_paths[i] for i in val_indices]
        
        logger.info(f"\nSplit dataset into:")
        logger.info(f"Train: {len(train_dataset)} volumes")
        logger.info(f"Val: {len(val_dataset)} volumes")
        
        return train_dataset, val_dataset