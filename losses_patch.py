import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class WeightedDiceBCELoss(nn.Module):
    """
    Combined Dice and BCE loss with stronger weighting for tumor voxels.
    
    Args:
        pos_weight: Weight for positive (tumor) class in BCE loss
        dice_weight: Weight for Dice loss component
        bce_weight: Weight for BCE loss component
        smooth: Smoothing factor for Dice loss
    """
    def __init__(self, 
                pos_weight: float = 10.0,
                dice_weight: float = 1.0,
                bce_weight: float = 1.0,
                smooth: float = 1e-5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]),
            reduction='mean'
        )
    
    def forward(self, 
               logits: torch.Tensor,
               target: torch.Tensor,
               kidney_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            logits: Raw model output [B, C, D, H, W]
            target: Ground truth mask [B, 1, D, H, W]
            kidney_mask: Optional kidney ROI mask [B, 1, D, H, W]
        """
        # Apply kidney mask if provided
        if kidney_mask is not None:
            # Ensure kidney mask is float and has same shape as input
            kidney_mask = kidney_mask.float()
            if kidney_mask.shape != target.shape:
                kidney_mask = F.interpolate(
                    kidney_mask,
                    size=target.shape[2:],
                    mode='nearest'
                )
            valid_pixels = kidney_mask > 0
        else:
            valid_pixels = torch.ones_like(target, dtype=torch.bool)
        
        # Get probabilities
        probs = torch.sigmoid(logits)
        
        # Compute BCE loss only on valid pixels
        bce_loss = self.bce(
            logits * valid_pixels,
            target * valid_pixels
        )
        
        # Compute Dice loss
        # Flatten predictions and targets for valid pixels only
        flat_probs = (probs * valid_pixels).view(probs.size(0), -1)
        flat_target = (target * valid_pixels).view(target.size(0), -1)
        
        intersection = (flat_probs * flat_target).sum(dim=1)
        union = flat_probs.sum(dim=1) + flat_target.sum(dim=1)
        
        # Handle empty masks
        valid_dice = union > 0
        if not valid_dice.any():
            # Return only BCE loss if no valid pixels
            return self.bce_weight * bce_loss
            
        dice = (2. * intersection[valid_dice] + self.smooth) / (union[valid_dice] + self.smooth)
        dice_loss = (1 - dice).mean()
        
        # Combine losses
        total_loss = (self.bce_weight * bce_loss + 
                     self.dice_weight * dice_loss)
        
        # For monitoring
        with torch.no_grad():
            n_tumor = target.sum().item()
            n_total = valid_pixels.sum().item()
            tumor_ratio = n_tumor / (n_total + 1e-8)
            if tumor_ratio < 0.01:  # Less than 1% tumor
                print(f"\nWarning: Very low tumor ratio in batch: {tumor_ratio:.4%}")
                print(f"Tumor voxels: {n_tumor}, Total valid voxels: {n_total}")
        
        return total_loss

    def log_stats(self, 
                logits: torch.Tensor, 
                target: torch.Tensor, 
                kidney_mask: Optional[torch.Tensor] = None):
        """Log detailed statistics about predictions and loss components"""
        with torch.no_grad():
            # Get probabilities
            probs = torch.sigmoid(logits)
            
            # Apply kidney mask
            if kidney_mask is not None:
                valid_pixels = kidney_mask > 0
                probs = probs * valid_pixels
                target = target * valid_pixels
            else:
                valid_pixels = torch.ones_like(target, dtype=torch.bool)
            
            # Basic stats
            n_tumor = target.sum().item()
            n_total = valid_pixels.sum().item()
            tumor_ratio = n_tumor / (n_total + 1e-8)
            
            # Prediction stats
            pred_tumor = (probs > 0.5).sum().item()
            true_pos = ((probs > 0.5) * target).sum().item()
            
            stats = {
                'tumor_ratio': tumor_ratio,
                'tumor_voxels': n_tumor,
                'total_voxels': n_total,
                'predicted_tumor': pred_tumor,
                'true_positives': true_pos,
                'mean_prob': probs[valid_pixels].mean().item(),
                'max_prob': probs[valid_pixels].max().item()
            }
            
            return stats