import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

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
        # Use reduction='none' so we can mask manually
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
            logits: Raw model output [B, 1, D, H, W]
            target: Ground truth mask [B, 1, D, H, W]
            kidney_mask: Optional kidney ROI mask [B, 1, D, H, W]
        """
        B, C, D, H, W = logits.shape
        device = logits.device

        if kidney_mask is not None:
            valid = (kidney_mask > 0).float()
        else:
            valid = torch.ones_like(target, device=device)

        # Compute BCE per-voxel (reduction='none')
        bce_raw = self.bce(logits, target)   # shape [B,1,D,H,W]

        # Mask out non-kidney voxels
        bce_masked = bce_raw * valid

        # Sum + normalize by #valid voxels
        n_valid = valid.sum()
        if n_valid > 0:
            bce_loss = bce_masked.sum() / (n_valid + 1e-6)
        else:
            bce_loss = torch.tensor(0.0, device=device)

        # Compute Dice terms on the masked region
        probs = torch.sigmoid(logits)
        probs_flat = (probs * valid).view(B, -1)
        target_flat = (target * valid).view(B, -1)

        intersection = (probs_flat * target_flat).sum(dim=1)     # [B]
        union = probs_flat.sum(dim=1) + target_flat.sum(dim=1)  # [B]

        # Only compute Dice loss for volumes where union>0
        mask_nonzero = union > 0
        if mask_nonzero.any():
            dice_score = (2.0 * intersection[mask_nonzero] + self.smooth) / (union[mask_nonzero] + self.smooth)
            dice_loss = (1.0 - dice_score).mean()
        else:
            dice_loss = torch.tensor(0.0, device=device)

        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss

        # Debug warning if tumor ratio is very low
        with torch.no_grad():
            n_tumor_voxels = target.sum()
            tumor_ratio = n_tumor_voxels.item() / (n_valid.item() + 1e-6)
            if tumor_ratio < 0.01:  # Less than 1% tumor
                print(f"\n[Loss Warning] Low tumor ratio: {tumor_ratio:.4%}")
                print(f"Tumor voxels: {n_tumor_voxels.item()}, Valid voxels: {n_valid.item()}")

        return total_loss

    def log_stats(self, 
                logits: torch.Tensor, 
                target: torch.Tensor, 
                kidney_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Log detailed statistics about predictions and loss components"""
        with torch.no_grad():
            device = logits.device
            
            # Get probabilities
            probs = torch.sigmoid(logits)
            
            # Apply kidney mask
            if kidney_mask is not None:
                valid = (kidney_mask > 0).float()
                probs = probs * valid
                target = target * valid
            else:
                valid = torch.ones_like(target, device=device)
            
            # Basic stats
            n_valid = valid.sum().item()
            n_tumor = target.sum().item()
            tumor_ratio = n_tumor / (n_valid + 1e-6)
            
            # Prediction stats at 0.5 threshold
            pred_mask = (probs > 0.5).float()
            n_pred_tumor = pred_mask.sum().item()
            true_pos = (pred_mask * target).sum().item()
            
            # Get probability stats in valid region
            valid_probs = probs[valid > 0]
            mean_prob = valid_probs.mean().item() if len(valid_probs) > 0 else 0
            max_prob = valid_probs.max().item() if len(valid_probs) > 0 else 0
            
            return {
                'tumor_ratio': tumor_ratio,
                'tumor_voxels': n_tumor,
                'valid_voxels': n_valid,
                'predicted_tumor': n_pred_tumor,
                'true_positives': true_pos,
                'mean_prob': mean_prob,
                'max_prob': max_prob,
                'dice_score': 2 * true_pos / (n_pred_tumor + n_tumor + 1e-6)
            }