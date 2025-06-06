import torch
import torch.nn.functional as F
import random
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, List
from pathlib import Path

class KiTS23PatchDataset(Dataset):
    """
    Dataset wrapper that samples patches from full volumes with a focus on tumor regions.
    Only requires CT and tumor mask (no kidney segmentation).
    """
    def __init__(self, 
                full_dataset: Dataset,
                patch_size: Tuple[int, int, int] = (64, 128, 128),
                tumor_only_prob: float = 0.7,
                debug: bool = False):
        super().__init__()
        self.base = full_dataset
        self.patch_size = patch_size
        self.tumor_only_prob = tumor_only_prob
        self.debug = debug
        
        # Set random seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        # For each volume, precompute tumor coordinates
        print("Precomputing tumor coordinates for efficient sampling...")
        self.tumor_indices = []
        total_tumor_voxels = 0
        volumes_with_tumor = 0
        
        for idx in range(len(full_dataset)):
            # Get masks from base dataset
            data = full_dataset[idx]
            if len(data) != 3:  # Still expect 3 items, but will ignore kidney mask
                raise ValueError(f"Base dataset must return (image, tumor_mask, _), got {len(data)} items")
            
            image, tumor_mask, _ = data  # Ignore kidney mask
            
            if not isinstance(tumor_mask, torch.Tensor):
                raise ValueError(f"Expected torch.Tensor tumor mask, got {type(tumor_mask)}")
            
            # Find tumor coordinates
            coords = (tumor_mask > 0).nonzero(as_tuple=False)
            coords = coords[:, 1:]  # Drop batch dimension
            self.tumor_indices.append(coords)
            
            # Track statistics
            n_tumor = len(coords)
            total_tumor_voxels += n_tumor
            if n_tumor > 0:
                volumes_with_tumor += 1
            
            if debug and idx < 5:
                print(f"\nVolume {idx}:")
                print(f"  Shape: {tumor_mask.shape}")
                print(f"  Total voxels: {tumor_mask.numel()}")
                print(f"  Tumor voxels: {n_tumor}")
                print(f"  Tumor percentage: {100 * n_tumor / tumor_mask.numel():.4f}%")
        
        print(f"\nDataset statistics:")
        print(f"Total volumes: {len(full_dataset)}")
        print(f"Volumes with tumor: {volumes_with_tumor}")
        print(f"Total tumor voxels: {total_tumor_voxels}")
        print(f"Average tumor voxels per volume: {total_tumor_voxels / len(full_dataset):.1f}")
        print(f"Tumor-centered patch probability: {tumor_only_prob:.1%}")
        print(f"Patch size: {patch_size}")

    def _extract_patch(self, 
                     volume: torch.Tensor,
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
            kidney_patch: [1, D, H, W] ones tensor (for compatibility)
        """
        # Get full volume data
        data = self.base[idx]
        if len(data) != 3:
            raise ValueError(f"Base dataset returned {len(data)} items, expected 3")
        image, tumor_mask, _ = data  # Ignore kidney mask
        
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
            Z, Y, X = tumor_mask.shape[1:]
            cz = random.randint(self.patch_size[0]//2, Z - self.patch_size[0]//2) if Z > self.patch_size[0] else Z//2
            cy = random.randint(self.patch_size[1]//2, Y - self.patch_size[1]//2) if Y > self.patch_size[1] else Y//2
            cx = random.randint(self.patch_size[2]//2, X - self.patch_size[2]//2) if X > self.patch_size[2] else X//2
            center = [cz, cy, cx]
        
        # Extract patches centered at the chosen coordinates
        image_patch = self._extract_patch(image, center, pad_value=0)
        tumor_patch = self._extract_patch(tumor_mask, center, pad_value=0)
        
        # Create dummy kidney mask of all ones
        kidney_patch = torch.ones_like(tumor_patch)
        
        # Verify patch sizes
        assert image_patch.shape[1:] == self.patch_size, f"Wrong patch size: {image_patch.shape[1:]} vs {self.patch_size}"
        assert tumor_patch.shape[1:] == self.patch_size
        assert kidney_patch.shape[1:] == self.patch_size
        
        # Log detailed patch statistics in debug mode
        if self.debug:
            tumor_voxels = (tumor_patch > 0).sum().item()
            total_voxels = tumor_patch.numel()
            
            print(f"\nPatch from volume {idx}:")
            print(f"  Center: {center}")
            print(f"  Tumor centered: {use_tumor}")
            print(f"  Tumor voxels: {tumor_voxels}/{total_voxels} ({100 * tumor_voxels / total_voxels:.4f}%)")
        
        return image_patch, tumor_patch, kidney_patch

    def __len__(self):
        return len(self.base)

    @staticmethod
    def collate_fn(batch):
        """Custom collate function if needed"""
        return torch.utils.data.dataloader.default_collate(batch)