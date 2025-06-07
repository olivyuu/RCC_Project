import torch
import torch.nn.functional as F
import random
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, List, Optional, Union
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
                data_source: Union[str, Dataset],
                patch_size: Tuple[int, int, int] = (64, 128, 128),
                tumor_only_prob: float = 0.7,
                use_kidney_mask: bool = False,
                min_kidney_voxels: int = 100,
                kidney_patch_overlap: float = 0.2,
                debug: bool = False):
        """
        Args:
            data_source: Either path to .pt files or a dataset object
            patch_size: Size of patches to extract (D,H,W)
            tumor_only_prob: Probability of sampling tumor-centered patches
            use_kidney_mask: Whether to use kidney masking (Phase 2)
            min_kidney_voxels: Minimum kidney voxels required in patch
            kidney_patch_overlap: Required tumor-kidney overlap ratio
            debug: Enable debug output
        """
        # Handle both string paths and dataset objects
        if isinstance(data_source, str):
            self.data_dir = Path(data_source)
            # Find .pt files directly
            self.volume_paths = sorted(list(self.data_dir.glob("case_*.pt")))
            if not self.volume_paths:
                raise RuntimeError(f"No preprocessed volumes found in {self.data_dir}")
        else:
            # Use paths and root_dir from dataset object
            self.volume_paths = data_source.volume_paths
            self.data_dir = data_source.root_dir
        
        self.patch_size = patch_size
        self.tumor_only_prob = tumor_only_prob
        self.use_kidney_mask = use_kidney_mask
        self.min_kidney_voxels = min_kidney_voxels
        self.kidney_patch_overlap = kidney_patch_overlap
        self.debug = debug

        # Track sampling statistics
        self.total_patches = 0
        self.tumor_centered_patches = 0
        self.kidney_valid_patches = 0
        self.failed_attempts = 0
        self.rejected_kidney_patches = 0
        self.rejected_overlap_patches = 0
        
        print("Computing coordinate lists for patch sampling...")
        
        # Store indices for each volume
        self.tumor_indices = []      # All tumor voxels
        self.kidney_indices = []     # All kidney voxels (including cysts)
        self.kidney_bg_indices = []  # Kidney voxels without tumor
        self.volume_shapes = []      # Store shapes for fallback sampling
        
        # Track statistics
        total_tumor_voxels = 0
        total_kidney_voxels = 0
        volumes_with_tumor = 0
        volumes_with_kidney = 0
        self.empty_kidney_volumes = set()
        
        # Build coordinate lists only for volumes in this dataset
        for idx, vol_path in enumerate(self.volume_paths):
            # Load preprocessed data
            data = torch.load(vol_path)
            image = data['image']
            tumor = data['tumor']
            kidney = data['kidney']
            
            # Store volume shape for fallback sampling
            self.volume_shapes.append(image.shape[1:])
            
            # Get all tumor coordinates regardless of kidney
            tumor_coords = (tumor > 0).nonzero(as_tuple=False)[:, 1:]
            
            if self.use_kidney_mask:
                # Include tumor regions in kidney mask
                full_kidney = (kidney > 0) | (tumor > 0)  # Include tumor in kidney mask
                # Get all kidney voxels (including tumor)
                kidney_coords = full_kidney.nonzero(as_tuple=False)[:, 1:]
                # Get kidney background (no tumor)
                kidney_bg = full_kidney & (tumor == 0)  # Update to use full kidney
                bg_coords = kidney_bg.nonzero(as_tuple=False)[:, 1:]
                
                if len(kidney_coords) == 0:
                    logger.warning(f"Volume {idx} has no kidney voxels - will use random sampling")
                    self.empty_kidney_volumes.add(idx)
            else:
                # Phase 1: Just use tumor coordinates
                kidney_coords = tumor_coords  # Placeholder
                bg_coords = tumor_coords      # Placeholder
            
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
        if self.use_kidney_mask:
            print(f"Kidney overlap threshold: {kidney_patch_overlap:.1%}")

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
                    logger.debug(f"Rejecting patch: only {n_kidney} kidney voxels (min: {self.min_kidney_voxels})")
                self.rejected_kidney_patches += 1
                return False
                
            # For tumor-centered patches, check tumor-kidney overlap
            if tumor_patch is not None:
                tumor_in_kidney = ((tumor_patch > 0) & (kidney_patch > 0)).sum()
                total_tumor = (tumor_patch > 0).sum()
                if total_tumor > 0:
                    overlap = tumor_in_kidney / total_tumor
                    if overlap < self.kidney_patch_overlap:
                        if self.debug:
                            logger.debug(f"Rejecting tumor patch: only {overlap:.1%} tumor-kidney overlap (min: {self.kidney_patch_overlap:.1%})")
                        self.rejected_overlap_patches += 1
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

        self.total_patches += 1
        use_tumor = (len(self.tumor_indices[idx]) > 0 and 
                    random.random() < self.tumor_only_prob)
        if use_tumor:
            self.tumor_centered_patches += 1
        
        # Handle volumes with no kidney in Phase 2
        if self.use_kidney_mask and idx in self.empty_kidney_volumes:
            center = self._get_random_center(idx)
            image_patch = self._extract_patch(image, center, pad_value=0)
            tumor_patch = self._extract_patch(tumor, center, pad_value=0)
            kidney_patch = torch.zeros_like(tumor_patch)
            if self.debug:
                logger.debug(f"Using random center for empty kidney volume {idx}")
            return image_patch, tumor_patch, kidney_patch
        
        max_attempts = 10
        for attempt in range(max_attempts):
            if use_tumor:
                # Select random tumor voxel (full tumor mask)
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
                # Create combined kidney mask including tumor
                full_kidney = (kidney > 0) | (tumor > 0)  # Add this line
                kidney_patch = self._extract_patch(full_kidney.float(), center, pad_value=0)
                # Validate patch
                if self._validate_kidney_patch(kidney_patch, tumor_patch if use_tumor else None):
                    self.kidney_valid_patches += 1
                    break
            else:
                kidney_patch = torch.ones_like(tumor_patch)
                break
                
        else:  # No valid patch found after max attempts
            self.failed_attempts += 1
            logger.warning(f"Could not find valid patch in volume {idx} after {max_attempts} attempts")
            # Return centered patch as fallback
            center = [d//2 for d in self.volume_shapes[idx]]
            image_patch = self._extract_patch(image, center, pad_value=0)
            tumor_patch = self._extract_patch(tumor, center, pad_value=0)
            if self.use_kidney_mask:
                # Use combined kidney mask in fallback too
                full_kidney = (kidney > 0) | (tumor > 0)  # Add this line
                kidney_patch = self._extract_patch(full_kidney.float(), center, pad_value=0)
            else:
                kidney_patch = torch.ones_like(tumor_patch)
        
        # Log sampling statistics periodically
        if self.total_patches % 1000 == 0:
            self._log_sampling_stats()
        
        # Verify patch sizes
        assert image_patch.shape[1:] == self.patch_size, f"Wrong patch size: {image_patch.shape[1:]} vs {self.patch_size}"
        assert tumor_patch.shape[1:] == self.patch_size
        assert kidney_patch.shape[1:] == self.patch_size
        
        # Log detailed patch statistics in debug mode
        if self.debug:
            tumor_voxels = (tumor_patch > 0).sum().item()
            total_voxels = tumor_patch.numel()
            
            stats = {
                'center': center,
                'tumor_centered': use_tumor,
                'tumor_voxels': tumor_voxels,
                'total_voxels': total_voxels,
                'tumor_ratio': tumor_voxels / total_voxels
            }
            
            if self.use_kidney_mask:
                kidney_voxels = (kidney_patch > 0).sum().item()
                stats.update({
                    'kidney_voxels': kidney_voxels,
                    'kidney_ratio': kidney_voxels / total_voxels
                })
                
                if use_tumor:
                    overlap = ((tumor_patch > 0) & (kidney_patch > 0)).sum().item()
                    stats['tumor_kidney_overlap'] = overlap / max(tumor_voxels, 1)
            
            logger.debug(f"Patch statistics for volume {idx}:")
            for k, v in stats.items():
                if isinstance(v, float):
                    logger.debug(f"  {k}: {v:.4f}")
                else:
                    logger.debug(f"  {k}: {v}")
        
        return image_patch, tumor_patch, kidney_patch

    def _log_sampling_stats(self):
        """Log detailed sampling statistics"""
        if self.total_patches == 0:
            logger.info("\nNo patches sampled yet")
            return
            
        logger.info("\nPatch sampling statistics:")
        logger.info(f"Total patches sampled: {self.total_patches}")
        
        tumor_ratio = 0 if self.total_patches == 0 else (self.tumor_centered_patches / self.total_patches)
        logger.info(f"Tumor-centered patches: {self.tumor_centered_patches} ({100*tumor_ratio:.1f}%)")
        
        if self.use_kidney_mask:
            kidney_ratio = 0 if self.total_patches == 0 else (self.kidney_valid_patches / self.total_patches)
            logger.info(f"Kidney-valid patches: {self.kidney_valid_patches} ({100*kidney_ratio:.1f}%)")
            
            tumor_reject_ratio = 0 if self.tumor_centered_patches == 0 else (
                self.rejected_kidney_patches / self.tumor_centered_patches)
            logger.info(f"Rejected kidney patches: {self.rejected_kidney_patches} ({100*tumor_reject_ratio:.1f}% of tumor patches)")
            
            overlap_reject_ratio = 0 if self.tumor_centered_patches == 0 else (
                self.rejected_overlap_patches / self.tumor_centered_patches)
            logger.info(f"Rejected overlap patches: {self.rejected_overlap_patches} ({100*overlap_reject_ratio:.1f}% of tumor patches)")
        
        fail_ratio = 0 if self.total_patches == 0 else (self.failed_attempts / self.total_patches)
        logger.info(f"Failed attempts: {self.failed_attempts} ({100*fail_ratio:.1f}%)")

    def __len__(self):
        return len(self.volume_paths)