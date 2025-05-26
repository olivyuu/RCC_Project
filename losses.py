import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Tuple

class BoundaryLoss(nn.Module):
    """
    Boundary loss to emphasize tumor edges
    Based on: https://arxiv.org/abs/1812.07032
    """
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        print(f"\nBoundaryLoss forward:")
        print(f"Input pred shape: {pred.shape}")
        print(f"Input target shape: {target.shape}")
        
        # Get probability map
        pred = pred.softmax(dim=1)
        print(f"After softmax shape: {pred.shape}")
        
        # For binary case, we only care about tumor class (channel 1)
        if pred.shape[1] == 2:
            pred = pred[:, 1:] 
        
        # Ensure target is in correct format
        if len(target.shape) == len(pred.shape) - 1:
            target = target.unsqueeze(1)
            
        target = (target > 0).float()
        
        # Interpolate pred to match target size
        if pred.shape[-3:] != target.shape[-3:]:
            pred = F.interpolate(pred, size=target.shape[-3:], mode='trilinear', align_corners=False)
            
        print(f"Processed shapes - pred: {pred.shape}, target: {target.shape}")
        
        # Calculate gradients
        grad_pred = self._compute_gradient(pred)
        grad_target = self._compute_gradient(target)
        print(f"Gradient shapes - pred: {grad_pred.shape}, target: {grad_target.shape}")
        
        # Calculate boundary loss
        boundary_loss = F.mse_loss(grad_pred, grad_target)
        return boundary_loss

    def _compute_gradient(self, x):
        print(f"Computing gradient for shape: {x.shape}")
        
        # Apply padding first to ensure consistent output size
        padded = F.pad(x, (1, 1, 1, 1, 1, 1), mode='replicate')
        print(f"After padding shape: {padded.shape}")
        
        # Compute gradients in xyz directions
        grad_x = torch.abs(F.conv3d(padded[:, :, :, 1:-1, 1:-1], self._get_sobel_kernel('x').to(x.device)))
        grad_y = torch.abs(F.conv3d(padded[:, :, :, 1:-1, 1:-1], self._get_sobel_kernel('y').to(x.device)))
        grad_z = torch.abs(F.conv3d(padded[:, :, :, 1:-1, 1:-1], self._get_sobel_kernel('z').to(x.device)))
        
        print(f"Gradient components - x: {grad_x.shape}, y: {grad_y.shape}, z: {grad_z.shape}")
        
        grad = (grad_x + grad_y + grad_z) / 3.0
        print(f"Combined gradient shape: {grad.shape}")
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
        else:  # z direction, make sure it's properly 3D
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
    """
    Focal Tversky Loss for improved small tumor segmentation
    """
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Get probability map
        pred = pred.softmax(dim=1)
        
        # For binary case, focus on tumor class
        if pred.shape[1] == 2:
            pred = pred[:, 1:]
            
        # Ensure target is in correct format
        if len(target.shape) == len(pred.shape) - 1:
            target = target.unsqueeze(1)
        
        target = (target > 0).float()
        
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
    def __init__(self, bce_kwargs: dict = None, soft_dice_kwargs: dict = None, 
                 weight_ce: float = 1.0, weight_dice: float = 1.0,
                 weight_boundary: float = 0.5, weight_focal_tversky: float = 0.5):
        super().__init__()
        if soft_dice_kwargs is None:
            soft_dice_kwargs = {'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}
        if bce_kwargs is None:
            bce_kwargs = {}

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_boundary = weight_boundary
        self.weight_focal_tversky = weight_focal_tversky
        
        self.ce = RobustCrossEntropyLoss(**bce_kwargs)
        self.dc = SoftDiceLoss(**soft_dice_kwargs)
        self.boundary = BoundaryLoss()
        self.focal_tversky = FocalTverskyLoss()

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Combined loss function
        """
        dc_loss = self.dc(net_output, target) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) if self.weight_ce != 0 else 0
        boundary_loss = self.boundary(net_output, target) if self.weight_boundary != 0 else 0
        focal_tversky_loss = self.focal_tversky(net_output, target) if self.weight_focal_tversky != 0 else 0
        
        result = (self.weight_ce * ce_loss + 
                 self.weight_dice * dc_loss +
                 self.weight_boundary * boundary_loss + 
                 self.weight_focal_tversky * focal_tversky_loss)
                 
        return result

class RobustCrossEntropyLoss(nn.Module):
    def __init__(self, reduction='none'):
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if input.shape[-3:] != target.shape[-3:]:
            input = F.interpolate(
                input,
                size=target.shape[-3:],
                mode='trilinear',
                align_corners=False
            )
        
        target = target.long()
        if len(target.shape) == len(input.shape) - 1:
            target = target.unsqueeze(1)
        elif len(target.shape) < len(input.shape) - 1:
            for _ in range(len(input.shape) - len(target.shape) - 1):
                target = target.unsqueeze(1)
        
        target_one_hot = torch.zeros_like(input)
        target_one_hot.scatter_(1, target, 1)
        
        log_softmax = F.log_softmax(input, dim=1)
        loss = -(target_one_hot * log_softmax).sum(dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss.mean()

class SoftDiceLoss(nn.Module):
    def __init__(self, batch_dice=True, smooth=1e-5, do_bg=False):
        super().__init__()
        self.batch_dice = batch_dice
        self.smooth = smooth
        self.do_bg = do_bg
        
    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if net_output.shape[-3:] != target.shape[-3:]:
            net_output = F.interpolate(
                net_output,
                size=target.shape[-3:],
                mode='trilinear',
                align_corners=False
            )
            
        shp_x = net_output.shape
        target = target.long()
        
        target_one_hot = torch.zeros_like(net_output)
        target_indices = target.long()
        if len(target_indices.shape) < len(target_one_hot.shape):
            target_indices = target_indices.unsqueeze(1)
            
        target_one_hot.scatter_(1, target_indices, 1)
        
        pc = net_output.softmax(1)
        
        if not self.do_bg:
            target_one_hot = target_one_hot[:, 1:, ...]
            pc = pc[:, 1:, ...]
        
        if self.batch_dice:
            if len(pc.shape) > 3:
                pc = pc.flatten(start_dim=2)
                target_one_hot = target_one_hot.flatten(start_dim=2)
            else:
                pc = pc.reshape(shp_x[0], shp_x[1], -1)
                target_one_hot = target_one_hot.reshape(shp_x[0], shp_x[1], -1)
            
            tp = (pc * target_one_hot).sum(-1)
            fp = pc.sum(-1) - tp
            fn = target_one_hot.sum(-1) - tp
            
            dice = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
            loss = (1 - dice).mean()
        else:
            raise NotImplementedError("Only batch dice supported")
            
        return loss