import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from pathlib import Path

class KiTS23Augmenter:
    def __init__(self, config):
        self.config = config
        self.rotation_angle = config.rotation_angle
        self.scale_range = config.scale_range
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Augmentation device: {self.device}")
        
    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Only move to GPU during processing if CUDA is available
        if torch.cuda.is_available():
            image = image.to(self.device)
            mask = mask.to(self.device)
        
        # Random rotation
        if torch.rand(1).item() < 0.8:  # Using torch.rand instead of np.random
            angle = torch.rand(1).item() * (self.rotation_angle[1] - self.rotation_angle[0]) + self.rotation_angle[0]
            image = self._rotate_3d(image, angle)
            mask = self._rotate_3d(mask, angle, is_mask=True)
        
        # Random scaling
        if torch.rand(1).item() < 0.5:
            scale = torch.rand(1).item() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
            image = self._scale_3d(image, scale)
            mask = self._scale_3d(mask, scale, is_mask=True)
        
        # Random Gaussian noise (directly on GPU)
        if torch.rand(1).item() < 0.2:
            noise = torch.randn_like(image) * 0.1
            image = image + noise
        
        # Random brightness adjustment
        if torch.rand(1).item() < 0.2:
            brightness = (torch.rand(1).to(self.device) * 0.3 + 0.85)  # Range [0.85, 1.15]
            image = image * brightness
        
        # Ensure mask remains binary
        mask = (mask > 0.5).float()

        # Ensure tensors are on CPU before returning
        if image.device.type == 'cuda':
            image = image.cpu()
        if mask.device.type == 'cuda':
            mask = mask.cpu()
        return image, mask
    
    def _rotate_3d(self, volume: torch.Tensor, angle: float, is_mask: bool = False) -> torch.Tensor:
        """Rotate 3D volume around z-axis using torch operations."""
        # Convert angle to radians
        theta = torch.tensor(angle * np.pi / 180).to(self.device)
        
        # Create rotation matrix
        # Create rotation matrix (3x3)
        rotation = torch.tensor([
            [torch.cos(theta), -torch.sin(theta), 0],
            [torch.sin(theta), torch.cos(theta), 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.device)

        # Add zero translation column to make it 3x4 for affine_grid
        zero_translation = torch.zeros(3, 1, dtype=torch.float32, device=self.device)
        rot_matrix = torch.cat([rotation, zero_translation], dim=1) # Shape becomes [3, 4]
        
        # Use grid_sample for rotation
        # affine_grid expects shape (N, 3, 4)
        grid = F.affine_grid(rot_matrix.unsqueeze(0), volume.unsqueeze(0).size(), align_corners=True)
        mode = 'nearest' if is_mask else 'bilinear'
        rotated = F.grid_sample(volume.unsqueeze(0), grid, mode=mode, align_corners=True)
        
        return rotated.squeeze(0)
    
    def _scale_3d(self, volume: torch.Tensor, scale: float, is_mask: bool = False) -> torch.Tensor:
        """Scale 3D volume using torch operations."""
        # Ensure minimum size
        min_size = 16
        
        # Handle input dimensions
        if len(volume.shape) == 3:  # D,H,W
            volume = volume.unsqueeze(0).unsqueeze(0)  # Add B,C dimensions
            squeeze_dims = 2
        elif len(volume.shape) == 4:  # C,D,H,W or B,D,H,W
            volume = volume.unsqueeze(0) if volume.shape[0] < 4 else volume.unsqueeze(1)
            squeeze_dims = 1
        elif len(volume.shape) == 5:  # B,C,D,H,W
            squeeze_dims = 0
        else:
            raise ValueError(f"Unexpected volume shape: {volume.shape}")
            
        # Calculate scaled shape for spatial dimensions only
        orig_shape = volume.shape
        scaled_shape = [max(min_size, int(s * scale)) for s in orig_shape[-3:]]
        mode = 'nearest' if is_mask else 'trilinear'
        
        # Perform interpolation
        scaled = F.interpolate(
            volume,
            size=scaled_shape,
            mode=mode,
            align_corners=None if is_mask else True
        )
        
        # Remove extra dimensions if needed
        for _ in range(squeeze_dims):
            scaled = scaled.squeeze(0)
            
        return scaled