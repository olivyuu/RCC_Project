import os
import torch
import numpy as np
from torch.utils.data import Dataset, random_split
from pathlib import Path
import logging
from typing import Optional, Tuple, Dict, List
from tqdm import tqdm

logger = logging.getLogger(__name__)

class KiTS23VolumeDataset(Dataset):
    """Dataset for loading preprocessed KiTS23 volumes and segmentations"""
    
    def __init__(self, 
                root_dir: str,
                config,
                preprocess: bool = False,
                phase: str = 'train',
                device: Optional[torch.device] = None):
        """
        Args:
            root_dir: Path to preprocessed .pt files
            config: Configuration object
            preprocess: Not used, kept for compatibility
            phase: 'train' or 'val' for different augmentation
            device: Device to load tensors to (default: None = CPU)
        """
        self.root_dir = Path(root_dir)
        self.config = config
        self.phase = phase
        self.device = device
        
        # Find all preprocessed volumes
        self.volume_paths = sorted(list(self.root_dir.glob("case_*.pt")))
        if not self.volume_paths:
            raise RuntimeError(f"No preprocessed volumes found in {self.root_dir}")
            
        logger.info(f"Found {len(self.volume_paths)} .pt files in: {self.root_dir}")
        
        # Cache volume statistics
        self.stats = self._compute_dataset_statistics()
        self._log_dataset_info()

    def _compute_dataset_statistics(self) -> Dict:
        """Compute and cache dataset statistics"""
        stats = {
            'n_volumes': len(self.volume_paths),
            'n_tumor': 0,
            'n_kidney': 0,
            'tumor_volumes': [],
            'kidney_volumes': [],
            'shapes': [],
            'spacings': []
        }
        
        logger.info("Computing dataset statistics...")
        for path in tqdm(self.volume_paths):
            try:
                data = torch.load(path)
                tumor_voxels = data['tumor'].sum().item()
                kidney_voxels = data['kidney'].sum().item()
                
                stats['n_tumor'] += tumor_voxels
                stats['n_kidney'] += kidney_voxels
                stats['tumor_volumes'].append(tumor_voxels)
                stats['kidney_volumes'].append(kidney_voxels)
                stats['shapes'].append(data['original_shape'])
                stats['spacings'].append(data['original_spacing'])
                
            except Exception as e:
                logger.error(f"Error loading {path}: {str(e)}")
                
        return stats

    def _log_dataset_info(self):
        """Log dataset statistics"""
        logger.info("\nDataset Statistics:")
        logger.info(f"Total volumes: {self.stats['n_volumes']}")
        logger.info(f"Total tumor voxels: {self.stats['n_tumor']}")
        logger.info(f"Total kidney voxels: {self.stats['n_kidney']}")
        logger.info(f"Average tumor voxels per volume: {self.stats['n_tumor'] / self.stats['n_volumes']:.1f}")
        logger.info(f"Average kidney voxels per volume: {self.stats['n_kidney'] / self.stats['n_volumes']:.1f}")
        
        if self.config.debug:
            # Log detailed stats for first few volumes
            for idx in range(min(3, len(self.volume_paths))):
                data = torch.load(self.volume_paths[idx])
                logger.debug(f"\nVolume {idx}:")
                logger.debug(f"  Shape: {data['image'].shape}")
                logger.debug(f"  Original shape: {data['original_shape']}")
                logger.debug(f"  Original spacing: {data['original_spacing']}")
                logger.debug(f"  Tumor voxels: {data['tumor'].sum().item()}")
                logger.debug(f"  Kidney voxels: {data['kidney'].sum().item()}")
                if 'source' in data:
                    logger.debug(f"  Source: {data['source']}")

    def get_volume_info(self, idx: int) -> Dict:
        """Get metadata for a volume"""
        data = torch.load(self.volume_paths[idx])
        return {
            'case_id': data.get('case_id', f"case_{idx:05d}"),
            'original_shape': data['original_shape'],
            'original_spacing': data['original_spacing'],
            'original_affine': data.get('original_affine', None),
            'tumor_voxels': self.stats['tumor_volumes'][idx],
            'kidney_voxels': self.stats['kidney_volumes'][idx]
        }

    def get_train_val_splits(self, 
                          val_ratio: float = 0.2,
                          seed: int = 42) -> Tuple[Dataset, Dataset]:
        """Create train/validation split"""
        val_size = int(len(self) * val_ratio)
        train_size = len(self) - val_size
        
        # Create new datasets for each split
        train_dataset = KiTS23VolumeDataset(
            self.root_dir,
            self.config,
            phase='train',
            device=self.device
        )
        val_dataset = KiTS23VolumeDataset(
            self.root_dir,
            self.config, 
            phase='val',
            device=self.device
        )
        
        # Split indices
        indices = list(range(len(self)))
        generator = torch.Generator().manual_seed(seed)
        train_indices, val_indices = random_split(indices, [train_size, val_size], generator=generator)
        
        # Assign paths to splits
        train_dataset.volume_paths = [self.volume_paths[i] for i in train_indices]
        val_dataset.volume_paths = [self.volume_paths[i] for i in val_indices]
        
        logger.info(f"Split dataset into {len(train_dataset)} train and {len(val_dataset)} validation volumes")
        return train_dataset, val_dataset

    def __len__(self):
        return len(self.volume_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load preprocessed volume data
        
        Returns:
            image: [1,D,H,W] Preprocessed CT volume
            tumor: [1,D,H,W] Tumor mask 
            kidney: [1,D,H,W] Kidney mask from KiTS23
        """
        try:
            # Load preprocessed data
            data = torch.load(self.volume_paths[idx])
            
            # Extract required tensors
            image = data['image']    # CT volume
            tumor = data['tumor']    # Tumor segmentation  
            kidney = data['kidney']  # KiTS23 kidney segmentation
            
            # Verify shapes match
            if not (image.shape == tumor.shape == kidney.shape):
                raise ValueError(
                    f"Shape mismatch in volume {idx}: "
                    f"image {image.shape}, tumor {tumor.shape}, kidney {kidney.shape}"
                )
            
            # Verify data format
            if not all(isinstance(x, torch.Tensor) for x in [image, tumor, kidney]):
                raise TypeError(f"Non-tensor data in volume {idx}")
                
            if not all(x.dim() == 4 for x in [image, tumor, kidney]):
                raise ValueError(f"Wrong tensor dimensions in volume {idx}")
                
            if not all(x.shape[0] == 1 for x in [image, tumor, kidney]):
                raise ValueError(f"Missing channel dimension in volume {idx}")
            
            # Move to device if specified
            if self.device is not None:
                image = image.to(self.device)
                tumor = tumor.to(self.device)
                kidney = kidney.to(self.device)
                
            return image, tumor, kidney
            
        except Exception as e:
            logger.error(f"Error loading volume {self.volume_paths[idx]}: {str(e)}")
            raise