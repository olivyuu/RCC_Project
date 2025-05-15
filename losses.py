import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Tuple

class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs: dict = None, ce_kwargs: dict = None, weight_ce: float = 1.0, weight_dice: float = 1.0):
        super().__init__()
        if soft_dice_kwargs is None:
            soft_dice_kwargs = {'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}
        if ce_kwargs is None:
            ce_kwargs = {'reduction': 'none'}

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(**soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        net_output: (batch_size, num_classes, ...)
        target: (batch_size, ...)
        """
        dc_loss = self.dc(net_output, target) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) if self.weight_ce != 0 else 0
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

class RobustCrossEntropyLoss(nn.Module):
    def __init__(self, reduction='none'):
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        input: (batch_size, num_classes, ...)
        target: (batch_size, ...)
        """
        # Ensure inputs have the same spatial dimensions
        if input.shape[-3:] != target.shape[-3:]:
            input = F.interpolate(
                input,
                size=target.shape[-3:],
                mode='trilinear',
                align_corners=False
            )
        
        # Convert target to long and handle dimensions
        target = target.long()
        if len(target.shape) == len(input.shape) - 1:
            target = target.unsqueeze(1)
        elif len(target.shape) < len(input.shape) - 1:
            for _ in range(len(input.shape) - len(target.shape) - 1):
                target = target.unsqueeze(1)
        
        # Create one hot target with correct shape
        target_one_hot = torch.zeros_like(input)
        target_one_hot.scatter_(1, target, 1)
        
        # Compute cross entropy loss
        log_softmax = F.log_softmax(input, dim=1)
        loss = -(target_one_hot * log_softmax).sum(dim=1)
        
        # Always reduce to scalar for backprop
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss.mean()  # Default to mean reduction

class SoftDiceLoss(nn.Module):
    def __init__(self, batch_dice=True, smooth=1e-5, do_bg=False):
        super().__init__()
        self.batch_dice = batch_dice
        self.smooth = smooth
        self.do_bg = do_bg
        
    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Ensure net_output and target have the same spatial dimensions
        if net_output.shape[-3:] != target.shape[-3:]:
            net_output = F.interpolate(
                net_output,
                size=target.shape[-3:],
                mode='trilinear',
                align_corners=False
            )
            
        shp_x = net_output.shape
        target = target.long()
        
        # Ensure target has the right shape for one-hot encoding
        if len(target.shape) == len(net_output.shape) - 1:
            target = target.unsqueeze(1)
        
        # Create tensor for one-hot encoding with same shape as net_output
        target_one_hot = torch.zeros_like(net_output)
        
        # Ensure all tensors have the same dimensions before scatter
        target_indices = target.long()
        if len(target_indices.shape) < len(target_one_hot.shape):
            target_indices = target_indices.unsqueeze(1)
            
        target_one_hot.scatter_(1, target_indices, 1)
        
        pc = net_output.softmax(1)
        
        if not self.do_bg:
            # Remove background class
            target_one_hot = target_one_hot[:, 1:, ...]
            pc = pc[:, 1:, ...]
        
        if self.batch_dice:
            # Handle 3D volume dimensions by properly flattening
            if len(pc.shape) > 3:  # For 3D volumes
                # Reshape keeping batch and class dims, flattening spatial dims
                spatial_size = int(torch.prod(torch.tensor(pc.shape[2:])))
                pc = pc.view(shp_x[0], shp_x[1], spatial_size)
                target_one_hot = target_one_hot.view(shp_x[0], shp_x[1], spatial_size)
            else:
                # Original behavior for 2D
                pc = pc.reshape(shp_x[0], shp_x[1], -1)
                target_one_hot = target_one_hot.reshape(shp_x[0], shp_x[1], -1)
            
            tp = (pc * target_one_hot).sum(-1)
            fp = pc.sum(-1) - tp
            fn = target_one_hot.sum(-1) - tp
            
            dice = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
            # Ensure scalar output
            loss = (1 - dice).mean()
        else:
            raise NotImplementedError("Only batch dice supported")
            
        return loss