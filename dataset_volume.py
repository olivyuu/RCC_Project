import torch
from pathlib import Path 
from typing import Tuple, Optional, List
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm import tqdm

class KiTS23VolumeDataset(Dataset):
    """Dataset for loading preprocessed KiTS23 volumes"""
    
    def __init__(self, 
                root_dir: str,
                config: object,
                preprocess: bool = True):
        """
        Args:
            root_dir: Path to preprocessed volumes directory
            config: Configuration object
            preprocess: Whether to preprocess volumes (or load preprocessed)
        """
        self.root_dir = Path(root_dir)
        self.config = config
        self.preprocess = preprocess
        
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.root_dir}")
            
        # Find all preprocessed .pt files
        self.volume_paths = sorted([
            f for f in self.root_dir.glob('*.pt')
            if f.name.startswith('case_') and f.name.endswith('_volume.pt')
        ])
        
        if not self.volume_paths:
            raise RuntimeError(f"No preprocessed volumes found in {self.root_dir}")
            
        print(f"Found {len(self.volume_paths)} preprocessed volumes")

    def __len__(self) -> int:
        return len(self.volume_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load a preprocessed volume
        
        Returns:
            tuple of (image, tumor_mask, kidney_mask) tensors
            shape: [1, D, H, W] for each
        """
        volume_path = self.volume_paths[idx]
        try:
            data = torch.load(volume_path)
            
            # Handle different data formats
            if isinstance(data, (tuple, list)) and len(data) == 3:
                # (image, tumor_mask, kidney_mask) format
                image, tumor_mask, kidney_mask = data
            elif isinstance(data, dict):
                # Dictionary format
                if all(k in data for k in ['image', 'tumor', 'kidney']):
                    image = data['image'] 
                    tumor_mask = data['tumor']
                    kidney_mask = data['kidney']
                else:
                    raise ValueError(f"Unknown dictionary keys in {volume_path}: {list(data.keys())}")
            else:
                raise ValueError(f"Unknown data format in {volume_path}: {type(data)}")
            
            # Ensure all tensors are proper shape and type
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            if len(tumor_mask.shape) == 3:
                tumor_mask = tumor_mask.unsqueeze(0)
            if len(kidney_mask.shape) == 3:
                kidney_mask = kidney_mask.unsqueeze(0)
                
            image = image.float()
            tumor_mask = tumor_mask.float()
            kidney_mask = kidney_mask.float()
            
            # Verify shapes match
            if not (image.shape == tumor_mask.shape == kidney_mask.shape):
                raise ValueError(f"Shape mismatch in {volume_path}: "
                            f"image {image.shape}, tumor {tumor_mask.shape}, "
                            f"kidney {kidney_mask.shape}")
                
            return image, tumor_mask, kidney_mask
            
        except Exception as e:
            print(f"Error loading {volume_path}: {str(e)}")
            raise

    def get_sample_id(self, idx: int) -> str:
        """Get case ID from volume path"""
        return self.volume_paths[idx].stem.split('_')[1]  # 'case_00123_volume' -> '00123'

    @staticmethod
    def collate_fn(batch):
        """Custom collate to handle different sized volumes"""
        # Find max dimensions
        max_d = max(x[0].shape[-3] for x in batch)
        max_h = max(x[0].shape[-2] for x in batch)
        max_w = max(x[0].shape[-1] for x in batch)
        
        # Pad each sample to max size
        padded_batch = []
        for image, tumor, kidney in batch:
            d, h, w = image.shape[-3:]
            
            # Calculate padding
            pad_d = max_d - d
            pad_h = max_h - h 
            pad_w = max_w - w
            
            # Pad all tensors
            image_pad = F.pad(image, (0, pad_w, 0, pad_h, 0, pad_d))
            tumor_pad = F.pad(tumor, (0, pad_w, 0, pad_h, 0, pad_d))
            kidney_pad = F.pad(kidney, (0, pad_w, 0, pad_h, 0, pad_d))
            
            padded_batch.append((image_pad, tumor_pad, kidney_pad))
        
        # Stack along batch dimension
        images, tumors, kidneys = zip(*padded_batch)
        return (torch.stack(images), 
                torch.stack(tumors),
                torch.stack(kidneys))