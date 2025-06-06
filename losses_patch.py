import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Union, Tuple

class WeightedDiceBCELoss(nn.Module):
    """Combined Dice and BCE loss with stronger weighting for tumor voxels"""
    def __init__(self, 
                pos_weight: float = 10.0,
                dice_weight: float = 1.0,
                bce_weight: float = 1.0,
                smooth: float = 1e-5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        
        # Use reduction='none' for per-voxel loss
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]),
            reduction='none'
        )
    
    def forward(self, 
               logits: torch.Tensor,
               target: torch.Tensor,
               kidney_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            logits: Model output [B,1,D,H,W]
            target: Ground truth mask [B,1,D,H,W]
            kidney_mask: Optional mask tensor (unused, kept for compatibility)
        """
        # Compute BCE loss (reduction='none')
        bce_raw = self.bce(logits, target)   # [B,1,D,H,W]
        bce_loss = bce_raw.mean()  # Average over all voxels
        
        # Compute Dice loss
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)  # [B,N]
        target_flat = target.view(target.size(0), -1)  # [B,N]
        
        intersection = (probs_flat * target_flat).sum(dim=1)     # [B]
        union = probs_flat.sum(dim=1) + target_flat.sum(dim=1)  # [B]
        
        # Compute Dice score and loss
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = (1.0 - dice_score).mean()
        
        # Combine losses
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss

        return total_loss

    def log_stats(self, 
                logits: torch.Tensor,
                target: torch.Tensor,
                kidney_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Log detailed statistics about predictions and loss components"""
        with torch.no_grad():
            # Get probabilities
            probs = torch.sigmoid(logits)
            
            # Basic stats
            n_voxels = target.numel()
            n_tumor = target.sum().item()
            tumor_ratio = n_tumor / n_voxels
            
            # Prediction stats at 0.5 threshold
            pred_mask = (probs > 0.5).float()
            n_pred_tumor = pred_mask.sum().item()
            true_pos = (pred_mask * target).sum().item()
            
            # Get probability stats
            valid_probs = probs.view(-1)
            mean_prob = valid_probs.mean().item()
            max_prob = valid_probs.max().item()
            
            return {
                'tumor_ratio': tumor_ratio,
                'tumor_voxels': n_tumor,
                'predicted_tumor': n_pred_tumor,
                'true_positives': true_pos,
                'mean_prob': mean_prob,
                'max_prob': max_prob,
                'dice_score': 2 * true_pos / (n_pred_tumor + n_tumor + 1e-6)
            }