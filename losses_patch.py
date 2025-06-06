import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Union, Tuple

class WeightedDiceBCELoss(nn.Module):
    """
    Combined Dice and BCE loss with stronger weighting for tumor voxels.
    Supports both Phase 1 (no masking) and Phase 2 (kidney masking) modes.
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
        Compute loss with optional kidney masking (Phase 2)
        
        Args:
            logits: Model output [B,1,D,H,W]
            target: Ground truth mask [B,1,D,H,W]
            kidney_mask: Optional kidney mask for Phase 2 [B,1,D,H,W]
                        If None or all ones, falls back to Phase 1 behavior
        """
        # Check if we're in Phase 1 (no mask) or Phase 2 (kidney mask)
        use_mask = kidney_mask is not None and not bool((kidney_mask == 1).all())
        
        # Phase 1: Standard BCE + Dice over all voxels
        if not use_mask:
            # Compute BCE loss
            bce_raw = self.bce(logits, target)
            bce_loss = bce_raw.mean()
            
            # Compute Dice loss
            probs = torch.sigmoid(logits)
            probs_flat = probs.view(probs.size(0), -1)
            target_flat = target.view(target.size(0), -1)
            
            intersection = (probs_flat * target_flat).sum(dim=1)
            union = probs_flat.sum(dim=1) + target_flat.sum(dim=1)
            
            dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss = (1.0 - dice_score).mean()
        
        # Phase 2: Masked BCE + Dice only inside kidney
        else:
            # Get kidney mask as float for multiplication
            mask = kidney_mask.float()
            
            # Compute masked BCE loss
            bce_raw = self.bce(logits, target)
            bce_masked = bce_raw * mask
            n_valid = mask.sum() + self.smooth
            bce_loss = bce_masked.sum() / n_valid
            
            # Compute masked Dice loss
            probs = torch.sigmoid(logits)
            
            # Apply mask to both predictions and targets
            probs_masked = probs * mask
            target_masked = target * mask
            
            # Flatten for Dice calculation
            probs_flat = probs_masked.view(probs.size(0), -1)
            target_flat = target_masked.view(target.size(0), -1)
            
            intersection = (probs_flat * target_flat).sum(dim=1)
            union = probs_flat.sum(dim=1) + target_flat.sum(dim=1)
            
            dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss = (1.0 - dice_score).mean()
        
        # Combine losses with weights
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        return total_loss

    def log_stats(self, 
                logits: torch.Tensor,
                target: torch.Tensor,
                kidney_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Log detailed statistics about predictions and loss components"""
        with torch.no_grad():
            # Check Phase 1 vs Phase 2
            use_mask = kidney_mask is not None and not bool((kidney_mask == 1).all())
            
            # Get probabilities
            probs = torch.sigmoid(logits)
            pred_mask = (probs > 0.5).float()
            
            if use_mask:
                # Phase 2: Calculate stats only inside kidney
                mask = kidney_mask.float()
                n_kidney = mask.sum().item()
                
                # Get masked predictions and targets
                probs_masked = probs * mask
                target_masked = target * mask
                pred_masked = pred_mask * mask
                
                # Calculate statistics
                n_voxels = n_kidney
                n_tumor = target_masked.sum().item()
                n_pred_tumor = pred_masked.sum().item()
                true_pos = (pred_masked * target_masked).sum().item()
                
                # Get probability stats for kidney voxels only
                valid_probs = probs_masked[mask > 0]
                mean_prob = valid_probs.mean().item() if len(valid_probs) > 0 else 0
                max_prob = valid_probs.max().item() if len(valid_probs) > 0 else 0
                
                # Calculate masked BCE loss
                bce_raw = self.bce(logits, target)
                bce_masked = bce_raw * mask
                bce_loss = bce_masked.sum().item() / (n_kidney + self.smooth)
                
            else:
                # Phase 1: Calculate stats over all voxels
                n_voxels = target.numel()
                n_tumor = target.sum().item()
                n_pred_tumor = pred_mask.sum().item()
                true_pos = (pred_mask * target).sum().item()
                
                # Get probability stats
                valid_probs = probs.view(-1)
                mean_prob = valid_probs.mean().item()
                max_prob = valid_probs.max().item()
                
                # Calculate BCE loss
                bce_raw = self.bce(logits, target)
                bce_loss = bce_raw.mean().item()
            
            # Calculate Dice score
            dice_score = 2 * true_pos / (n_pred_tumor + n_tumor + self.smooth)
            dice_loss = 1.0 - dice_score
            
            # Calculate total loss
            total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
            
            stats = {
                'loss': total_loss,
                'bce_loss': bce_loss,
                'dice_loss': dice_loss,
                'tumor_ratio': n_tumor / n_voxels if n_voxels > 0 else 0,
                'tumor_voxels': n_tumor,
                'predicted_tumor': n_pred_tumor,
                'true_positives': true_pos,
                'mean_prob': mean_prob,
                'max_prob': max_prob,
                'dice_score': dice_score
            }
            
            if use_mask:
                stats.update({
                    'kidney_voxels': n_kidney,
                    'kidney_ratio': n_kidney / target.numel(),
                    'tumor_in_kidney_ratio': n_tumor / (n_kidney + self.smooth)
                })
                
            return stats