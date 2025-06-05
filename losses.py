import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Tuple
from dataclasses import dataclass

@dataclass
class LossWeights:
    """Class to store and update loss weights during training"""
    ce: float = 1.0  # Start with high CE weight
    dice: float = 0.5  # Start with lower Dice weight
    boundary: float = 0.2
    focal_tversky: float = 0.5

    def update_for_epoch(self, epoch: int):
        """Gradually adjust weights as training progresses"""
        if epoch < 5:  # First 5 epochs: emphasize CE
            self.ce = 1.0
            self.dice = 0.5
            self.focal_tversky = 0.5
        elif 5 <= epoch < 10:  # Next 5 epochs: transition
            self.ce = 0.5
            self.dice = 1.0
            self.focal_tversky = 0.75
        else:  # Later epochs: tumor-focused weights
            self.ce = 0.2
            self.dice = 2.0
            self.focal_tversky = 1.0

class SoftDiceLoss(nn.Module):
    def __init__(self, batch_dice=True, smooth=1e-5, do_bg=False):
        super().__init__()
        self.batch_dice = batch_dice
        self.smooth = smooth
        self.do_bg = do_bg

    def forward(self, net_output: torch.Tensor,
                target: torch.Tensor,
                kidney_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss only within kidney regions
        Args:
            net_output: [B, 2, D, H, W] logits
            target: [B, D, H, W] or [B, 1, D, H, W] tumor masks
            kidney_mask: [B, D, H, W] or [B, 1, D, H, W] kidney masks
        """
        # Get probability map
        pc = net_output.softmax(dim=1)
        
        # For binary case, focus on tumor class
        if not self.do_bg and pc.shape[1] == 2:
            pc = pc[:, 1:]
        
        # Ensure target and kidney_mask are proper shape
        if len(target.shape) == len(pc.shape) - 1:
            target = target.unsqueeze(1)
        target = target.float()
        
        if len(kidney_mask.shape) == len(pc.shape) - 1:
            kidney_mask = kidney_mask.unsqueeze(1)
        mask = (kidney_mask > 0).float()
        
        # Apply kidney mask to both predictions and targets
        pc = pc * mask
        target = target * mask
        
        # Interpolate if needed
        if pc.shape[-3:] != target.shape[-3:]:
            pc = F.interpolate(pc, size=target.shape[-3:],
                             mode='trilinear', align_corners=False)
            
        # Flatten kidney regions
        pc_flat = pc.flatten(2)
        tgt_flat = target.flatten(2)
        
        # Calculate Dice score
        intersection = (pc_flat * tgt_flat).sum(dim=2)
        union = pc_flat.sum(dim=2) + tgt_flat.sum(dim=2)
        dice_score = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score.mean()
        
        return dice_loss

    def compute_soft_dice_score(self, net_output: torch.Tensor, 
                              target: torch.Tensor,
                              kidney_mask: torch.Tensor) -> float:
        """Compute soft Dice score without thresholding"""
        with torch.no_grad():
            pc = net_output.softmax(dim=1)
            if pc.shape[1] == 2:
                pc = pc[:, 1:]
            
            if len(target.shape) == len(pc.shape) - 1:
                target = target.unsqueeze(1)
            
            if len(kidney_mask.shape) == len(pc.shape) - 1:
                kidney_mask = kidney_mask.unsqueeze(1)
            mask = (kidney_mask > 0).float()
            
            # Apply kidney mask
            pc = pc * mask
            target = target * mask
            
            pc_flat = pc.flatten(2)
            tgt_flat = target.flatten(2)
            
            intersection = (pc_flat * tgt_flat).sum(dim=2)
            union = pc_flat.sum(dim=2) + tgt_flat.sum(dim=2)
            
            dice_score = (2 * intersection + self.smooth) / (union + self.smooth)
            return dice_score.mean().item()

class RobustCrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, 
                target: torch.Tensor,
                kidney_mask: torch.Tensor) -> torch.Tensor:
        if input.shape[-3:] != target.shape[-3:]:
            input = F.interpolate(input, size=target.shape[-3:], 
                                mode='trilinear', align_corners=False)
        
        # Ensure target values are valid (0 or 1)
        target = (target > 0).long()
        
        # Handle dimensions
        if len(target.shape) == len(input.shape) - 1:
            target = target.unsqueeze(1)
            
        if len(kidney_mask.shape) == len(input.shape) - 1:
            kidney_mask = kidney_mask.unsqueeze(1)
        mask = (kidney_mask > 0).float()
        
        num_classes = input.shape[1]
        target_one_hot = torch.zeros_like(input)
        target = torch.clamp(target, 0, num_classes - 1)
        target_one_hot.scatter_(1, target, 1)
        
        # Apply kidney mask to one-hot target
        target_one_hot = target_one_hot * mask
        
        log_softmax = F.log_softmax(input, dim=1)
        loss = -(target_one_hot * log_softmax).sum(dim=1)
        
        # Apply kidney mask to loss
        loss = loss * mask.squeeze(1)
        
        # Average only over kidney voxels
        if self.reduction == 'mean':
            num_kidney_voxels = mask.sum()
            return loss.sum() / (num_kidney_voxels + 1e-5)
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class BoundaryLoss(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, pred: torch.Tensor, 
                target: torch.Tensor,
                kidney_mask: torch.Tensor) -> torch.Tensor:
        # Get probability map
        pred = pred.softmax(dim=1)
        
        # For binary case, focus on tumor class
        if pred.shape[1] == 2:
            pred = pred[:, 1:]
        
        # Ensure proper dimensions
        if len(target.shape) == len(pred.shape) - 1:
            target = target.unsqueeze(1)
        target = (target > 0).float()
        
        if len(kidney_mask.shape) == len(pred.shape) - 1:
            kidney_mask = kidney_mask.unsqueeze(1)
        mask = (kidney_mask > 0).float()
        
        # Apply kidney mask
        pred = pred * mask
        target = target * mask
        
        # Calculate gradients
        grad_pred = self._compute_gradient(pred)
        grad_target = self._compute_gradient(target)
        
        # Calculate boundary loss only within kidney
        boundary_loss = F.mse_loss(grad_pred * mask, grad_target * mask)
        return boundary_loss

    def _compute_gradient(self, x):
        device = x.device
        kernel_x = self._get_sobel_kernel('x').to(device)
        kernel_y = self._get_sobel_kernel('y').to(device)
        kernel_z = self._get_sobel_kernel('z').to(device)

        padded = F.pad(x, (1, 1, 1, 1, 1, 1), mode='reflect')

        grad_x = torch.abs(F.conv3d(padded, kernel_x, padding=(0, 1, 1)))
        grad_y = torch.abs(F.conv3d(padded, kernel_y, padding=(0, 1, 1)))
        grad_z = torch.abs(F.conv3d(padded, kernel_z, padding=(1, 1, 1)))
        
        grad = (grad_x + grad_y + grad_z) / 3.0
        return grad

    def _get_sobel_kernel(self, direction):
        if direction == 'x':
            kernel = torch.tensor([[[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]]])
        elif direction == 'y':
            kernel = torch.tensor([[[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]]])
        else:  # z direction
            kernel = torch.tensor([[[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 0]],
                                [[0, 1, 0],
                                [1, 2, 1],
                                [0, 1, 0]],
                                [[0, 0, 0],
                                [0, -1, 0],
                                [0, 0, 0]]])
        
        return kernel.unsqueeze(0).unsqueeze(0).float()

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, 
                target: torch.Tensor,
                kidney_mask: torch.Tensor) -> torch.Tensor:
        # Get probability map
        pred = pred.softmax(dim=1)
        
        # For binary case, focus on tumor class
        if pred.shape[1] == 2:
            pred = pred[:, 1:]
        
        # Ensure target and kidney_mask dimensions
        if len(target.shape) == len(pred.shape) - 1:
            target = target.unsqueeze(1)
        target = (target > 0).float()
        
        if len(kidney_mask.shape) == len(pred.shape) - 1:
            kidney_mask = kidney_mask.unsqueeze(1)
        mask = (kidney_mask > 0).float()
        
        # Apply kidney mask
        pred = pred * mask
        target = target * mask
        
        # Flatten predictions and targets
        pred = pred.flatten(2)
        target = target.flatten(2)
        
        # Calculate True Positives, False Positives, and False Negatives
        tp = (pred * target).sum(dim=2)
        fp = (pred * (1 - target)).sum(dim=2)
        fn = ((1 - pred) * target).sum(dim=2)
        
        # Calculate Tversky Index
        numerator = tp + self.smooth
        denominator = tp + self.alpha * fp + self.beta * fn + self.smooth
        tversky = numerator / denominator
        
        # Apply focal factor
        focal_tversky = torch.pow((1 - tversky), self.gamma)
        
        return focal_tversky.mean()

class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs: dict = None, soft_dice_kwargs: dict = None):
        super().__init__()
        if soft_dice_kwargs is None:
            soft_dice_kwargs = {'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}
        if bce_kwargs is None:
            bce_kwargs = {}

        self.weights = LossWeights()
        
        self.ce = RobustCrossEntropyLoss(**bce_kwargs)
        self.dc = SoftDiceLoss(**soft_dice_kwargs)
        self.boundary = BoundaryLoss()
        self.focal_tversky = FocalTverskyLoss()

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, kidney_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute combined loss with kidney masking
        Args:
            net_output: Model output logits
            target: Ground truth tumor mask
            kidney_mask: Optional kidney mask to focus loss computation
        """
        # Default kidney mask to all ones if not provided
        if kidney_mask is None:
            kidney_mask = torch.ones_like(target)
        
        dc_loss = self.dc(net_output, target, kidney_mask)
        ce_loss = self.ce(net_output, target, kidney_mask)
        boundary_loss = self.boundary(net_output, target, kidney_mask)
        focal_tversky_loss = self.focal_tversky(net_output, target, kidney_mask)
        
        return (self.weights.ce * ce_loss + 
                self.weights.dice * dc_loss +
                self.weights.boundary * boundary_loss + 
                self.weights.focal_tversky * focal_tversky_loss)

    def update_weights(self, epoch: int):
        """Update loss weights based on training epoch"""
        self.weights.update_for_epoch(epoch)