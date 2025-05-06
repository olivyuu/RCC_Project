import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
from typing import Dict, Tuple

class MetricsTracker:
    def __init__(self, log_dir: str = "runs"):
        self.writer = SummaryWriter(log_dir)
        
    def update_metrics(self, outputs: torch.Tensor, targets: torch.Tensor, 
                      loss: float, phase: str, epoch: int) -> Dict[str, float]:
        """Calculate and log metrics for each epoch."""
        with torch.no_grad():
            # Convert outputs to predictions
            if isinstance(outputs, list):  # Handle deep supervision
                outputs = outputs[0]  # Use main output
            predictions = torch.argmax(outputs, dim=1)
            
            # Calculate metrics
            dice = self.dice_coefficient(predictions, targets)
            sensitivity = self.sensitivity(predictions, targets)
            precision = self.precision(predictions, targets)
            
            # Log to tensorboard
            self.writer.add_scalar(f'{phase}/Loss', loss, epoch)
            self.writer.add_scalar(f'{phase}/Dice', dice, epoch)
            self.writer.add_scalar(f'{phase}/Sensitivity', sensitivity, epoch)
            self.writer.add_scalar(f'{phase}/Precision', precision, epoch)
            
            return {
                'loss': loss,
                'dice': dice,
                'sensitivity': sensitivity,
                'precision': precision
            }
    
    @staticmethod
    def dice_coefficient(pred: torch.Tensor, target: torch.Tensor) -> float:
        smooth = 1e-5
        intersection = (pred * target).sum()
        return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    @staticmethod
    def sensitivity(pred: torch.Tensor, target: torch.Tensor) -> float:
        smooth = 1e-5
        true_positives = (pred * target).sum()
        actual_positives = target.sum()
        return (true_positives + smooth) / (actual_positives + smooth)
    
    @staticmethod
    def precision(pred: torch.Tensor, target: torch.Tensor) -> float:
        smooth = 1e-5
        true_positives = (pred * target).sum()
        predicted_positives = pred.sum()
        return (true_positives + smooth) / (predicted_positives + smooth)