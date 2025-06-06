import multiprocessing as mp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler
import torch.cuda
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import numpy as np
from tqdm import tqdm
import signal
import sys
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from models.segmentation.model import SegmentationModel
from dataset_patch import KiTS23PatchDataset
from losses_patch import WeightedDiceBCELoss
from dataset_volume import KiTS23VolumeDataset

class PatchSegmentationTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Configure CUDA memory management
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.95)
            torch.backends.cudnn.benchmark = True
        
        # Create checkpoint directory
        self.config.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize model
        self.model = SegmentationModel(
            in_channels=2,  # CT and kidney mask channels
            out_channels=2,  # Two channels for background/tumor
            features=config.features
        ).to(self.device)
        
        # Training settings
        self.max_grad_norm = 5.0
        self.batch_size = 2
        self.tumor_only_prob = 0.7
        self.patch_size = (64, 128, 128)
        self.num_workers = 4
        self.start_epoch = 0
        self.current_epoch = 0
        self.best_val_dice = float('-inf')
        
        # Initialize training components
        self.criterion = WeightedDiceBCELoss(
            pos_weight=10.0,  # Higher weight for tumor voxels
            dice_weight=1.0,
            bce_weight=1.0
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=5e-5,  # Lower initial learning rate
            weight_decay=1e-5,
            eps=1e-8
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )
        
        self.scaler = GradScaler()
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        if config.resume_training:
            self._load_checkpoint()

    def _train_epoch(self, train_loader):
        """Run one epoch of training"""
        self.model.train()
        train_loss = 0
        valid_batches = 0
        total_norm = 0
        
        # Track statistics
        tumor_ratios = []
        mean_probs = []
        max_probs = []
        dice_scores = []
        skipped_batches = 0
        
        with tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}/{self.config.num_epochs}") as pbar:
            for batch_idx, (images, tumor_masks, kidney_masks) in enumerate(pbar):
                # Move data to device
                images = images.to(device=self.device)
                tumor_masks = tumor_masks.to(device=self.device)
                kidney_masks = kidney_masks.to(device=self.device)
                
                # Skip patches with no kidney voxels
                if kidney_masks.sum() == 0:
                    if self.config.debug:
                        print(f"\n[Warning] Skipping batch {batch_idx} (no kidney in patch)")
                    skipped_batches += 1
                    continue
                
                # Get CT and kidney channels
                ct_images = images[:, 0:1].float()
                kidney_inputs = kidney_masks.float()
                
                # Combine input channels
                inputs = torch.cat([ct_images, kidney_inputs], dim=1)
                
                # Forward pass
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, tumor_masks, kidney_masks)
                
                # Skip if loss is invalid
                if not torch.isfinite(loss):
                    print(f"\n[Warning] Skipping batch {batch_idx} (invalid loss: {loss.item()})")
                    skipped_batches += 1
                    continue
                
                # Backward pass with gradient clipping
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Compute metrics
                with torch.no_grad():
                    stats = self.criterion.log_stats(outputs, tumor_masks, kidney_masks)
                    tumor_ratios.append(stats['tumor_ratio'])
                    mean_probs.append(stats['mean_prob'])
                    max_probs.append(stats['max_prob'])
                    dice_scores.append(stats['dice_score'])
                    
                    if batch_idx % 50 == 0:
                        self._log_batch_stats(stats, batch_idx, self.current_epoch)
                    
                    train_loss += loss.item()
                    valid_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'tumor_ratio': f"{stats['tumor_ratio']:.4%}",
                    'dice': f"{stats['dice_score']:.4f}",
                    'grad_norm': f"{total_norm:.4f}"
                })
                
                # Clear cache periodically
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
        
        # Calculate epoch statistics
        if valid_batches > 0:
            avg_stats = {
                'loss': train_loss / valid_batches,
                'tumor_ratio': np.mean(tumor_ratios),
                'mean_prob': np.mean(mean_probs),
                'max_prob': np.mean(max_probs),
                'dice_score': np.mean(dice_scores)
            }
        else:
            print("\nWarning: No valid batches in epoch!")
            avg_stats = {
                'loss': float('inf'),
                'tumor_ratio': 0,
                'mean_prob': 0,
                'max_prob': 0,
                'dice_score': 0
            }
        
        if skipped_batches > 0:
            print(f"\nSkipped {skipped_batches} batches due to no kidney or invalid loss")
        
        return avg_stats['loss'], avg_stats

    def _validate(self, val_loader):
        """Validate on full volumes"""
        self.model.eval()
        val_loss = 0
        val_dice = 0
        valid_batches = 0
        
        with tqdm(val_loader, desc="Validating") as pbar:
            for images, tumor_masks, kidney_masks in pbar:
                # Move to device
                images = images.to(device=self.device)
                tumor_masks = tumor_masks.to(device=self.device)
                kidney_masks = kidney_masks.to(device=self.device)
                
                # Get inputs
                ct_images = images[:, 0:1].float()
                kidney_inputs = kidney_masks.float()
                inputs = torch.cat([ct_images, kidney_inputs], dim=1)
                
                try:
                    with torch.no_grad():
                        # Forward pass
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, tumor_masks, kidney_masks)
                        
                        # Skip invalid loss
                        if not torch.isfinite(loss):
                            print(f"\n[Warning] Skipping validation batch (invalid loss)")
                            continue
                            
                        # Get predictions
                        if isinstance(outputs, (list, tuple)):
                            outputs = outputs[-1]  # Take last output if list/tuple
                            
                        if outputs.shape[1] == 2:  # Two-channel output
                            outputs = outputs[:, 1:2]  # Take tumor channel
                            
                        if outputs.shape[2:] != tumor_masks.shape[2:]:
                            outputs = F.interpolate(
                                outputs,
                                size=tumor_masks.shape[2:],
                                mode='trilinear',
                                align_corners=False
                            )
                        
                        probs = torch.sigmoid(outputs)
                        preds = (probs > 0.5).float()
                        
                        # Calculate Dice only within kidney
                        dice = self._calculate_dice(preds, tumor_masks, kidney_masks)
                        
                        val_loss += loss.item()
                        val_dice += dice
                        valid_batches += 1
                        
                        pbar.set_postfix({
                            'loss': f"{loss.item():.4f}",
                            'dice': f"{dice:.4f}"
                        })
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\nCUDA OOM during validation. Skipping batch.")
                        torch.cuda.empty_cache()
                        continue
                    raise e
        
        return (val_loss / valid_batches if valid_batches > 0 else float('inf'),
                val_dice / valid_batches if valid_batches > 0 else 0)

    def _calculate_dice(self, 
                      pred: torch.Tensor,
                      target: torch.Tensor,
                      mask: torch.Tensor,
                      smooth: float = 1e-5) -> float:
        """Calculate Dice score for predictions"""
        # Apply mask
        pred = pred * mask
        target = target * mask
        
        # Calculate Dice
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice.item()

    def _log_batch_stats(self, stats: dict, batch_idx: int, epoch: int):
        """Log detailed batch statistics to tensorboard"""
        step = epoch * 1000 + batch_idx  # Unique step number
        self.writer.add_scalar('Batch/tumor_ratio', stats['tumor_ratio'], step)
        self.writer.add_scalar('Batch/true_positives', stats['true_positives'], step)
        self.writer.add_scalar('Batch/mean_prob', stats['mean_prob'], step)
        self.writer.add_scalar('Batch/max_prob', stats['max_prob'], step)
        self.writer.add_scalar('Batch/dice_score', stats['dice_score'], step)

    def _log_epoch(self, train_loss, train_stats, val_loss, val_dice, lr, epoch):
        """Log epoch-level metrics"""
        # Training metrics
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/val', val_loss, epoch)
        self.writer.add_scalar('Dice/val', val_dice, epoch)
        self.writer.add_scalar('LearningRate', lr, epoch)
        
        # Training statistics
        self.writer.add_scalar('Train/tumor_ratio', train_stats['tumor_ratio'], epoch)
        self.writer.add_scalar('Train/mean_prob', train_stats['mean_prob'], epoch)
        self.writer.add_scalar('Train/max_prob', train_stats['max_prob'], epoch)
        self.writer.add_scalar('Train/dice_score', train_stats['dice_score'], epoch)
        
        print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_stats['dice_score']:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")
        print(f"Learning Rate: {lr:.6f}")
        print(f"Mean tumor ratio: {train_stats['tumor_ratio']:.4%}")

    def _visualize_predictions(self, val_loader, epoch):
        """Visualize predictions on validation data"""
        self.model.eval()
        
        # Get first validation batch
        images, tumor_masks, kidney_masks = next(iter(val_loader))
        images = images.to(self.device)
        tumor_masks = tumor_masks.to(self.device)
        kidney_masks = kidney_masks.to(self.device)
        
        with torch.no_grad():
            # Get predictions
            ct_images = images[:, 0:1].float()
            kidney_inputs = kidney_masks.float()
            inputs = torch.cat([ct_images, kidney_inputs], dim=1)
            outputs = self.model(inputs)
            
            # Handle model output
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[-1]
            if outputs.shape[1] == 2:
                outputs = outputs[:, 1:2]
            if outputs.shape[2:] != tumor_masks.shape[2:]:
                outputs = F.interpolate(
                    outputs,
                    size=tumor_masks.shape[2:],
                    mode='trilinear',
                    align_corners=False
                )
                
            probs = torch.sigmoid(outputs)
            
            # Get middle slice
            slice_idx = probs.shape[2] // 2
            
            # Create figure
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            # Plot CT
            axes[0].imshow(ct_images[0, 0, slice_idx].cpu(), cmap='gray')
            axes[0].set_title('CT')
            
            # Plot kidney mask
            axes[1].imshow(kidney_masks[0, 0, slice_idx].cpu(), cmap='gray')
            axes[1].set_title('Kidney')
            
            # Plot ground truth tumor
            axes[2].imshow(tumor_masks[0, 0, slice_idx].cpu(), cmap='gray')
            axes[2].set_title('Ground Truth')
            
            # Plot prediction
            axes[3].imshow(probs[0, 0, slice_idx].cpu(), cmap='gray')
            axes[3].set_title(f'Prediction (Epoch {epoch+1})')
            
            # Save figure
            self.writer.add_figure('Predictions', fig, epoch)
            plt.close(fig)

    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_dice': self.best_val_dice,
            'metrics': metrics,
            'config': {
                'patch_size': self.patch_size,
                'batch_size': self.batch_size,
                'tumor_only_prob': self.tumor_only_prob
            }
        }
        
        latest_path = self.config.checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)
        
        if is_best:
            best_path = self.config.checkpoint_dir / f"best_model_dice_{metrics['dice']:.4f}.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model with Dice: {metrics['dice']:.4f}")

    def _load_checkpoint(self):
        """Load training checkpoint"""
        latest_path = self.config.checkpoint_dir / "latest.pth"
        if not latest_path.exists():
            print(f"No checkpoint found at {latest_path}")
            return
        
        print(f"Loading checkpoint: {latest_path}")
        checkpoint = torch.load(latest_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_dice = checkpoint['best_val_dice']
        
        # Load configuration if available
        if 'config' in checkpoint:
            self.patch_size = checkpoint['config']['patch_size']
            self.batch_size = checkpoint['config']['batch_size']
            self.tumor_only_prob = checkpoint['config']['tumor_only_prob']
        
        print(f"Resuming training from epoch {self.start_epoch}")