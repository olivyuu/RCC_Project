import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, List, Dict
from pathlib import Path
import logging
import numpy as np
import gc

logger = logging.getLogger(__name__)

class KiTS23VolumeDataset(Dataset):
    """Dataset for full-volume training with kidney masking support"""
    
    def __init__(self,
                data_dir: str,
                use_kidney_mask: bool = True,
                training: bool = True,
                debug: bool = False):
        """
        Args:
            data_dir: Path to preprocessed .pt files
            use_kidney_mask: Whether to return 2-channel input (CT + kidney)
            training: Whether this is training set (True) or validation (False)
            debug: Enable debug logging
        """
        self.data_dir = Path(data_dir) / "preprocessed_volumes"  # Match VolumePreprocessor output dir
        self.use_kidney_mask = use_kidney_mask
        self.training = training
        self.debug = debug
        
        # Find all preprocessed .pt files with correct naming pattern
        self.volume_paths = sorted(list(self.data_dir.glob("case_*.pt")))
        if not self.volume_paths:
            raise RuntimeError(f"No preprocessed volumes found in {self.data_dir}")
        
        logger.info(f"Found {len(self.volume_paths)} preprocessed volumes in {self.data_dir}")
        logger.info(f"Input channels: {2 if use_kidney_mask else 1} (CT{' + kidney mask' if use_kidney_mask else ''})")
        logger.info(f"Mode: {'Training' if training else 'Validation'}")
        logger.info("Using full volumes (no sliding windows)")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a volume"""
        try:
            # Load preprocessed tensors
            data = torch.load(self.volume_paths[idx])
            image = data['image']         # [1,D,H,W]
            tumor = data['tumor']         # [1,D,H,W]
            
            # Only load kidney mask if needed
            if self.use_kidney_mask:
                kidney = data['kidney']       # [1,D,H,W] combined mask (kidney+tumor+cyst)
                input_tensor = torch.cat([image, kidney], dim=0)  # [2,D,H,W]
                # Clean up
                del kidney
            else:
                input_tensor = image  # [1,D,H,W]
                
            # Clean up image tensor after concatenation
            del image
            
            return {
                'input': input_tensor,
                'target': tumor,
                'case_id': data['case_id'],
                'shape': data['original_shape'],
                'spacing': data['original_spacing']
            }
                
        finally:
            # Final cleanup
            torch.cuda.empty_cache()
            gc.collect()

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
        
        # Create train/val datasets with appropriate settings
        train_dataset = KiTS23VolumeDataset(
            data_dir=str(self.data_dir.parent),  # Remove preprocessed_volumes from path
            use_kidney_mask=self.use_kidney_mask,
            training=True,
            debug=self.debug
        )
        train_dataset.volume_paths = [self.volume_paths[i] for i in train_indices]
        
        val_dataset = KiTS23VolumeDataset(
            data_dir=str(self.data_dir.parent),  # Remove preprocessed_volumes from path
            use_kidney_mask=self.use_kidney_mask,
            training=False,
            debug=self.debug
        )
        val_dataset.volume_paths = [self.volume_paths[i] for i in val_indices]
        
        logger.info(f"\nSplit dataset into:")
        logger.info(f"Train: {len(train_dataset)} volumes")
        logger.info(f"Val: {len(val_dataset)} volumes")
        
        return train_dataset, val_dataset