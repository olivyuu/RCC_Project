import torch
import torch.nn.functional as F
import random
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, List, Optional
from pathlib import Path

class KiTS23PatchDataset(Dataset):
    """Dataset wrapper that samples patches with kidney-aware sampling for Phase 2"""
    def __init__(self, 
                full_dataset: Dataset,
                patch_size: Tuple[int, int, int] = (64, 128, 128),
                tumor_only_prob: float = 0.7,
                use_kidney_mask: bool = False,  # Phase 2: Enable kidney masking
                min_kidney_voxels: int = 100,  # Minimum kidney voxels for valid patch
                kidney_patch_overlap: float = 0.5,  # Minimum overlap with kidney
                debug: bool = False):
        super().__init__()
        self.base = full_dataset
        self.patch_size = patch_size
        self.tumor_only_prob = tumor_only_prob
        self.use_kidney_mask = use_kidney_mask
        self.min_kidney_voxels = min_kidney_voxels
        self.kidney_patch_overlap = kidney_patch_overlap
        self.debug = debug
        
        # Set random seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        print("Precomputing coordinates for efficient sampling...")
        
        # Store indices for each volume
        self.tumor_indices = []  # Tumor voxels that are also in kidney
        self.kidney_indices = []  # All kidney voxels
        self.kidney_bg_indices = []  # Kidney voxels without tumor
        
        # Track statistics
        total_tumor_voxels = 0
        total_kidney_voxels = 0
        volumes_with_tumor = 0
        volumes_with_kidney = 0
        
        for idx in range(len(full_dataset)):
            # Get masks
            data = full_dataset[idx]
            if len(data) != 3:
                raise ValueError(f"Base dataset must return (image, tumor_mask, kidney_mask), got {len(data)} items")
            
            _, tumor_mask, kidney_mask = data
            
            if not all(isinstance(x, torch.Tensor) for x in [tumor_mask, kidney_mask]):
                raise ValueError("Expected torch.Tensor masks")
            
            # Find tumor coordinates inside kidney
            if self.use_kidney_mask:
                tumor_and_kidney = (tumor_mask > 0) & (kidney_mask > 0)
                tumor_coords = tumor_and_kidney.nonzero(as_tuple=False)[:, 1:]  # Drop batch dim
                # Store all kidney coordinates for background sampling
                kidney_coords = (kidney_mask > 0).nonzero(as_tuple=False)[:, 1:]
                # Find kidney background (no tumor) coordinates
                kidney_bg = (kidney_mask > 0) & (tumor_mask == 0)
                bg_coords = kidney_bg.nonzero(as_tuple=False)[:, 1:]
            else:
                # Phase 1: Use only tumor mask
                tumor_coords = (tumor_mask > 0).nonzero(as_tuple=False)[:, 1:]
                kidney_coords = tumor_coords  # Placeholder
                bg_coords = tumor_coords  # Placeholder
            
            self.tumor_indices.append(tumor_coords)
            self.kidney_indices.append(kidney_coords)
            self.kidney_bg_indices.append(bg_coords)
            
            # Track statistics
            n_tumor = len(tumor_coords)
            n_kidney = len(kidney_coords)
            total_tumor_voxels += n_tumor
            total_kidney_voxels += n_kidney
            
            if n_tumor > 0:
                volumes_with_tumor += 1
            if n_kidney > 0:
                volumes_with_kidney += 1
            
            if debug and idx < 5:
                print(f"\nVolume {idx}:")
                print(f"  Shape: {tumor_mask.shape}")
                print(f"  Total voxels: {tumor_mask.numel()}")
                print(f"  Tumor voxels: {n_tumor}")
                if self.use_kidney_mask:
                    print(f"  Kidney voxels: {n_kidney}")
                    print(f"  Tumor in kidney voxels: {len(tumor_coords)}")
                    print(f"  Tumor percentage in kidney: {100 * len(tumor_coords) / max(n_kidney, 1):.4f}%")
                else:
                    print(f"  Tumor percentage: {100 * n_tumor / tumor_mask.numel():.4f}%")
        
        print(f"\nDataset statistics:")
        print(f"Total volumes: {len(full_dataset)}")
        print(f"Volumes with tumor: {volumes_with_tumor}")
        if self.use_kidney_mask:
            print(f"Volumes with kidney: {volumes_with_kidney}")
            print(f"Total kidney voxels: {total_kidney_voxels}")
            print(f"Average kidney voxels per volume: {total_kidney_voxels / len(full_dataset):.1f}")
        print(f"Total tumor voxels: {total_tumor_voxels}")
        print(f"Average tumor voxels per volume: {total_tumor_voxels / len(full_dataset):.1f}")
        print(f"Tumor-centered patch probability: {tumor_only_prob:.1%}")
        print(f"Patch size: {patch_size}")
        if self.use_kidney_mask:
            print(f"Minimum kidney voxels per patch: {min_kidney_voxels}")
            print(f"Required kidney overlap: {kidney_patch_overlap:.1%}")

    def _extract_patch(self, 
                     volume: torch.Tensor,
                     center: List[int],
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
        
        # Pad if necessary
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

    def _validate_kidney_patch(self, 
                            kidney_patch: torch.Tensor, 
                            tumor_patch: Optional[torch.Tensor] = None) -> bool:
        """
        Validate a patch based on kidney content criteria
        Returns: True if patch is valid, False if it should be resampled
        """
        if not self.use_kidney_mask:
            return True
            
        # Check minimum kidney voxels
        n_kidney = (kidney_patch > 0).sum().item()
        if n_kidney < self.min_kidney_voxels:
            if self.debug:
                print(f"Rejecting patch: only {n_kidney} kidney voxels (min: {self.min_kidney_voxels})")
            return False
            
        # For tumor-centered patches, check tumor-kidney overlap
        if tumor_patch is not None:
            tumor_in_kidney = ((tumor_patch > 0) & (kidney_patch > 0)).sum()
            total_tumor = (tumor_patch > 0).sum()
            if total_tumor > 0:
                overlap = tumor_in_kidney / total_tumor
                if overlap < self.kidney_patch_overlap:
                    if self.debug:
                        print(f"Rejecting tumor patch: only {overlap:.1%} tumor-kidney overlap (min: {self.kidney_patch_overlap:.1%})")
                    return False
                    
        return True

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            image_patch: [1, D, H, W] CT patch
            tumor_patch: [1, D, H, W] tumor mask patch
            kidney_patch: [1, D, H, W] kidney mask patch (or ones if Phase 1)
        """
        # Get full volume data
        data = self.base[idx]
        if len(data) != 3:
            raise ValueError(f"Base dataset returned {len(data)} items, expected 3")
        image, tumor_mask, kidney_mask = data
        
        # Decide whether to sample tumor-centered patch
        use_tumor = (len(self.tumor_indices[idx]) > 0 and 
                    random.random() < self.tumor_only_prob)
        
        max_attempts = 10
        for attempt in range(max_attempts):
            if use_tumor:
                # Select random tumor voxel (guaranteed to be in kidney in Phase 2)
                coords = self.tumor_indices[idx]
                center_idx = random.randint(0, len(coords) - 1)
                center = coords[center_idx].tolist()
            else:
                if self.use_kidney_mask:
                    # Select random kidney background voxel
                    coords = self.kidney_bg_indices[idx]
                    if len(coords) == 0:  # Fall back to any kidney voxel
                        coords = self.kidney_indices[idx]
                    center_idx = random.randint(0, len(coords) - 1)
                    center = coords[center_idx].tolist()
                else:
                    # Phase 1: Select random center
                    Z, Y, X = tumor_mask.shape[1:]
                    cz = random.randint(self.patch_size[0]//2, Z - self.patch_size[0]//2)
                    cy = random.randint(self.patch_size[1]//2, Y - self.patch_size[1]//2)
                    cx = random.randint(self.patch_size[2]//2, X - self.patch_size[2]//2)
                    center = [cz, cy, cx]
            
            # Extract patches
            image_patch = self._extract_patch(image, center, pad_value=0)
            tumor_patch = self._extract_patch(tumor_mask, center, pad_value=0)
            
            if self.use_kidney_mask:
                kidney_patch = self._extract_patch(kidney_mask, center, pad_value=0)
                # Validate patch
                if self._validate_kidney_patch(kidney_patch, tumor_patch if use_tumor else None):
                    break
            else:
                kidney_patch = torch.ones_like(tumor_patch)
                break
                
        else:  # No valid patch found after max attempts
            if self.debug:
                print(f"Warning: Could not find valid patch after {max_attempts} attempts")
            # Return centered patch as fallback
            Z, Y, X = tumor_mask.shape[1:]
            center = [Z//2, Y//2, X//2]
            image_patch = self._extract_patch(image, center, pad_value=0)
            tumor_patch = self._extract_patch(tumor_mask, center, pad_value=0)
            kidney_patch = self._extract_patch(kidney_mask, center, pad_value=0) if self.use_kidney_mask else torch.ones_like(tumor_patch)
        
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
            if self.use_kidney_mask:
                kidney_voxels = (kidney_patch > 0).sum().item()
                print(f"  Kidney voxels: {kidney_voxels}/{total_voxels} ({100 * kidney_voxels / total_voxels:.4f}%)")
                if use_tumor:
                    overlap = ((tumor_patch > 0) & (kidney_patch > 0)).sum().item()
                    print(f"  Tumor-kidney overlap: {overlap}/{tumor_voxels} ({100 * overlap / max(tumor_voxels, 1):.4f}%)")
        
        return image_patch, tumor_patch, kidney_patch

    def __len__(self):
        return len(self.base)

    @staticmethod
    def collate_fn(batch):
        """Custom collate function if needed"""
        return torch.utils.data.dataloader.default_collate(batch)