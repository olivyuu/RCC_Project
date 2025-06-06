import torch
import torch.nn.functional as F
import random
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def worker_init_fn(worker_id: int):
    """Initialize random seeds uniquely for each worker"""
    worker_seed = 42 + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

class KiTS23PatchDataset(Dataset):
    """Dataset wrapper that samples patches from preprocessed volumes"""
    
    def __init__(self, 
                data_dir: str,
                patch_size: Tuple[int, int, int] = (64, 128, 128),
                tumor_only_prob: float = 0.7,
                use_kidney_mask: bool = False,
                min_kidney_voxels: int = 100,
                kidney_patch_overlap: float = 0.5,
                debug: bool = False):
        """
        Args:
            data_dir: Directory containing preprocessed .pt files
            patch_size: Size of patches to extract (D,H,W)
            tumor_only_prob: Probability of sampling tumor-centered patches
            use_kidney_mask: Whether to use kidney masking (Phase 2)
            min_kidney_voxels: Minimum kidney voxels required in patch
            kidney_patch_overlap: Required tumor-kidney overlap ratio
            debug: Enable debug output
        """
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.tumor_only_prob = tumor_only_prob
        self.use_kidney_mask = use_kidney_mask
        self.min_kidney_voxels = min_kidney_voxels
        self.kidney_patch_overlap = kidney_patch_overlap
        self.debug = debug
        
        # Find all preprocessed volumes
        self.volume_paths = sorted(list(self.data_dir.glob("case_*.pt")))
        if not self.volume_paths:
            raise RuntimeError(f"No preprocessed volumes found in {self.data_dir}")
        
        print("Precomputing coordinates for efficient sampling...")
        
        # Store indices for each volume
        self.tumor_indices = []  # Tumor voxels that are also in kidney
        self.kidney_indices = []  # All kidney voxels
        self.kidney_bg_indices = []  # Kidney voxels without tumor
        self.volume_shapes = []  # Store shapes for fallback sampling
        
        # Track statistics
        total_tumor_voxels = 0
        total_kidney_voxels = 0
        volumes_with_tumor = 0
        volumes_with_kidney = 0
        self.empty_kidney_volumes = set()  # Track volumes with no kidney
        
        for idx, vol_path in enumerate(self.volume_paths):
            # Load preprocessed data
            data = torch.load(vol_path)
            image = data['image']
            tumor = data['tumor']
            kidney = data['kidney']
            
            # Store volume shape for fallback sampling
            self.volume_shapes.append(image.shape[1:])  # [D,H,W]
            
            # Find coordinates based on phase
            if self.use_kidney_mask:
                tumor_and_kidney = (tumor > 0) & (kidney > 0)
                tumor_coords = tumor_and_kidney.nonzero(as_tuple=False)[:, 1:]
                kidney_coords = (kidney > 0).nonzero(as_tuple=False)[:, 1:]
                kidney_bg = (kidney > 0) & (tumor == 0)
                bg_coords = kidney_bg.nonzero(as_tuple=False)[:, 1:]
                
                # Check for empty kidney
                if len(kidney_coords) == 0:
                    logger.warning(f"Volume {idx} has no kidney voxels - will use random sampling")
                    self.empty_kidney_volumes.add(idx)
            else:
                # Phase 1: Use only tumor mask
                tumor_coords = (tumor > 0).nonzero(as_tuple=False)[:, 1:]
                kidney_coords = tumor_coords  # Placeholder
                bg_coords = tumor_coords  # Placeholder
            
            self.tumor_indices.append(tumor_coords)
            self.kidney_indices.append(kidney_coords)
            self.kidney_bg_indices.append(bg_coords)
            
            # Track statistics
            n_tumor = len(tumor_coords)
            n_kidney = len(kidney_coords) if self.use_kidney_mask else n_tumor
            total_tumor_voxels += n_tumor
            total_kidney_voxels += n_kidney
            
            if n_tumor > 0:
                volumes_with_tumor += 1
            if n_kidney > 0 or not self.use_kidney_mask:
                volumes_with_kidney += 1
            
            if debug and idx < 5:
                print(f"\nVolume {idx}:")
                print(f"  Shape: {image.shape}")
                print(f"  Total voxels: {image.numel()}")
                print(f"  Tumor voxels: {n_tumor}")
                if self.use_kidney_mask:
                    print(f"  Kidney voxels: {n_kidney}")
                    print(f"  Tumor in kidney: {len(tumor_coords)}")
                    print(f"  Tumor percentage in kidney: {100 * len(tumor_coords) / max(n_kidney, 1):.4f}%")
                else:
                    print(f"  Tumor percentage: {100 * n_tumor / image.numel():.4f}%")
        
        print(f"\nDataset statistics:")
        print(f"Total volumes: {len(self.volume_paths)}")
        print(f"Volumes with tumor: {volumes_with_tumor}")
        if self.use_kidney_mask:
            print(f"Volumes with kidney: {volumes_with_kidney}")
            print(f"Volumes without kidney: {len(self.empty_kidney_volumes)}")
            print(f"Total kidney voxels: {total_kidney_voxels}")
            print(f"Average kidney voxels per volume: {total_kidney_voxels / len(self.volume_paths):.1f}")
        print(f"Total tumor voxels: {total_tumor_voxels}")
        print(f"Average tumor voxels per volume: {total_tumor_voxels / len(self.volume_paths):.1f}")
        print(f"Tumor-centered patch probability: {tumor_only_prob:.1%}")
        print(f"Patch size: {patch_size}")

    def _get_random_center(self, volume_idx: int) -> List[int]:
        """Get random center coordinates respecting patch bounds"""
        D, H, W = self.volume_shapes[volume_idx]
        return [
            random.randint(self.patch_size[0]//2, D - self.patch_size[0]//2),
            random.randint(self.patch_size[1]//2, H - self.patch_size[1]//2),
            random.randint(self.patch_size[2]//2, W - self.patch_size[2]//2)
        ]

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
        
        # Extract patch
        patch = volume[:, z0:z1, y0:y1, x0:x1]
        
        # Calculate padding if needed
        pad_z = (max(0, D//2 - cz), max(0, (cz + D//2) - Z))
        pad_y = (max(0, H//2 - cy), max(0, (cy + H//2) - Y))
        pad_x = (max(0, W//2 - cx), max(0, (cx + W//2) - X))
        
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
            
        with torch.no_grad():
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
        # Load preprocessed volume data
        data = torch.load(self.volume_paths[idx])
        image = data['image']
        tumor = data['tumor']
        kidney = data['kidney']
        
        # Decide whether to sample tumor-centered patch
        use_tumor = (len(self.tumor_indices[idx]) > 0 and 
                    random.random() < self.tumor_only_prob)
        
        # Handle volumes with no kidney in Phase 2
        if self.use_kidney_mask and idx in self.empty_kidney_volumes:
            center = self._get_random_center(idx)
            image_patch = self._extract_patch(image, center, pad_value=0)
            tumor_patch = self._extract_patch(tumor, center, pad_value=0)
            kidney_patch = torch.zeros_like(tumor_patch)  # No kidney present
            if self.debug:
                print(f"Using random center for empty kidney volume {idx}")
            return image_patch, tumor_patch, kidney_patch
        
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
                    if len(coords) == 0:  # No kidney - use random center
                        center = self._get_random_center(idx)
                    else:
                        center_idx = random.randint(0, len(coords) - 1)
                        center = coords[center_idx].tolist()
                else:
                    # Phase 1: Select random center
                    center = self._get_random_center(idx)
            
            # Extract patches
            image_patch = self._extract_patch(image, center, pad_value=0)
            tumor_patch = self._extract_patch(tumor, center, pad_value=0)
            
            if self.use_kidney_mask:
                kidney_patch = self._extract_patch(kidney, center, pad_value=0)
                # Validate patch
                if self._validate_kidney_patch(kidney_patch, tumor_patch if use_tumor else None):
                    break
            else:
                kidney_patch = torch.ones_like(tumor_patch)
                break
                
        else:  # No valid patch found after max attempts
            logger.warning(f"Could not find valid patch in volume {idx} after {max_attempts} attempts")
            # Return centered patch as fallback
            center = [d//2 for d in self.volume_shapes[idx]]
            image_patch = self._extract_patch(image, center, pad_value=0)
            tumor_patch = self._extract_patch(tumor, center, pad_value=0)
            kidney_patch = self._extract_patch(kidney, center, pad_value=0) if self.use_kidney_mask else torch.ones_like(tumor_patch)
        
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
        return len(self.volume_paths)

# For DataLoader worker initialization
__all__ = ['KiTS23PatchDataset', 'worker_init_fn']