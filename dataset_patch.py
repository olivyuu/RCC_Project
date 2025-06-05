import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path
import random

class KiTS23PatchDataset(Dataset):
    """
    Dataset wrapper that samples patches from full volumes with a focus on tumor regions.
    
    Args:
        full_dataset: Base KiTS23VolumeDataset that returns (image, tumor_mask, kidney_mask)
        patch_size: (D, H, W) size of patches to extract
        tumor_only_prob: Probability of sampling a patch centered on tumor (vs random)
        debug: Whether to print debug information about patches
    """
    def __init__(self, 
                full_dataset: Dataset,
                patch_size: Tuple[int, int, int] = (64, 128, 128),
                tumor_only_prob: float = 0.7,
                debug: bool = False):
        self.base = full_dataset
        self.patch_size = patch_size
        self.tumor_only_prob = tumor_only_prob
        self.debug = debug
        
        # For each volume, precompute the list of tumor voxel coordinates
        print("Precomputing tumor coordinates for efficient sampling...")
        self.tumor_indices = []
        total_tumor_voxels = 0
        
        for idx in range(len(full_dataset)):
            _, tumor_mask, _ = full_dataset[idx]
            coords = (tumor_mask > 0).nonzero(as_tuple=False)  # [N, 4] -> [batch, z, y, x]
            # Drop batch dimension since it's always 1
            coords = coords[:, 1:]  # [N, 3] -> [z, y, x]
            self.tumor_indices.append(coords)
            total_tumor_voxels += len(coords)
            
            if debug and idx < 5:
                print(f"Volume {idx}:")
                print(f"  Total voxels: {tumor_mask.numel()}")
                print(f"  Tumor voxels: {len(coords)}")
                print(f"  Tumor percentage: {100 * len(coords) / tumor_mask.numel():.4f}%")
        
        print(f"\nDataset statistics:")
        print(f"Total volumes: {len(full_dataset)}")
        print(f"Total tumor voxels: {total_tumor_voxels}")
        print(f"Average tumor voxels per volume: {total_tumor_voxels / len(full_dataset):.1f}")

    def __len__(self):
        return len(self.base)

    def _extract_patch(self, volume: torch.Tensor, 
                     center: Tuple[int, int, int],
                     pad_value: float = 0) -> torch.Tensor:
        """Extract a patch of self.patch_size centered at the given coordinates"""
        Z, Y, X = volume.shape[1:]  # Ignore channel dimension
        D, H, W = self.patch_size
        cz, cy, cx = center
        
        # Calculate patch bounds with edge clamping
        z0 = max(0, cz - D//2)
        z1 = min(Z, cz + D//2)
        y0 = max(0, cy - H//2)
        y1 = min(Y, cy + H//2)
        x0 = max(0, cx - W//2)
        x1 = min(X, cx + W//2)
        
        # Calculate required padding
        pad_z = (max(0, D//2 - cz), max(0, (cz + D//2) - Z))
        pad_y = (max(0, H//2 - cy), max(0, (cy + H//2) - Y))
        pad_x = (max(0, W//2 - cx), max(0, (cx + W//2) - X))
        
        # Extract patch
        patch = volume[:, z0:z1, y0:y1, x0:x1]
        
        # Pad if necessary to achieve target size
        if sum(pad_z + pad_y + pad_x) > 0:
            patch = F.pad(
                patch,
                pad=(pad_x[0], pad_x[1],  # Last dim (W)
                     pad_y[0], pad_y[1],  # Second-to-last dim (H)
                     pad_z[0], pad_z[1]),  # Third-to-last dim (D)
                mode='constant',
                value=pad_value
            )
        
        return patch

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            image_patch: [1, D, H, W] CT patch
            tumor_patch: [1, D, H, W] tumor mask patch
            kidney_patch: [1, D, H, W] kidney mask patch
        """
        # Get full volume data
        image, tumor_mask, kidney_mask = self.base[idx]
        Z, Y, X = tumor_mask.shape[1:]  # Ignore channel dimension
        
        # Decide whether to sample tumor-centered patch
        use_tumor = (len(self.tumor_indices[idx]) > 0 and 
                    random.random() < self.tumor_only_prob)
        
        if use_tumor:
            # Select random tumor voxel as center
            coords = self.tumor_indices[idx]
            center_idx = random.randint(0, len(coords) - 1)
            center = coords[center_idx].tolist()
        else:
            # Select random center within volume bounds
            cz = random.randint(self.patch_size[0]//2, Z - self.patch_size[0]//2) if Z > self.patch_size[0] else Z//2
            cy = random.randint(self.patch_size[1]//2, Y - self.patch_size[1]//2) if Y > self.patch_size[1] else Y//2
            cx = random.randint(self.patch_size[2]//2, X - self.patch_size[2]//2) if X > self.patch_size[2] else X//2
            center = [cz, cy, cx]
        
        # Extract patches centered at the chosen coordinates
        image_patch = self._extract_patch(image, center, pad_value=0)
        tumor_patch = self._extract_patch(tumor_mask, center, pad_value=0)
        kidney_patch = self._extract_patch(kidney_mask, center, pad_value=0)
        
        # Verify patch sizes
        assert image_patch.shape[1:] == self.patch_size, f"Wrong patch size: {image_patch.shape[1:]} vs {self.patch_size}"
        assert tumor_patch.shape[1:] == self.patch_size
        assert kidney_patch.shape[1:] == self.patch_size
        
        if self.debug:
            tumor_voxels = (tumor_patch > 0).sum().item()
            total_voxels = tumor_patch.numel()
            print(f"\nPatch from volume {idx}:")
            print(f"  Center: {center}")
            print(f"  Tumor centered: {use_tumor}")
            print(f"  Tumor voxels: {tumor_voxels}/{total_voxels}")
            print(f"  Tumor percentage: {100 * tumor_voxels / total_voxels:.4f}%")
        
        # Apply data augmentation here if needed
        
        return image_patch, tumor_patch, kidney_patch

    @staticmethod
    def collate_fn(batch):
        """Custom collate function if needed"""
        return torch.utils.data.dataloader.default_collate(batch)