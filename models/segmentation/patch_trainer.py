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
            in_channels=1,  # CT channel only
            out_channels=1,  # Single channel for tumor probability
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
            lr=5e-5,
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

    def train(self, dataset_path: str):
        """Main training loop"""
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass

        # Create training dataset
        full_dataset = KiTS23VolumeDataset(dataset_path, self.config, preprocess=self.config.preprocess)
        
        # Split into train/val
        val_size = int(len(full_dataset) * 0.2)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Wrap training data in patch sampler
        train_patch_dataset = KiTS23PatchDataset(
            train_dataset,
            patch_size=self.patch_size,
            tumor_only_prob=self.tumor_only_prob,
            debug=self.config.debug
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_patch_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True
        )

        print(f"\nTraining configuration:")
        print(f"Patch size: {self.patch_size}")
        print(f"Batch size: {self.batch_size}")
        print(f"Tumor sampling probability: {self.tumor_only_prob}")
        print(f"Training patches per epoch: {len(train_loader)}")
        print(f"Validation volumes: {len(val_dataset)}")

        try:
            signal.signal(signal.SIGINT, self._handle_interrupt)
            signal.signal(signal.SIGTERM, self._handle_interrupt)
            
            for epoch in range(self.start_epoch, self.config.num_epochs):
                self.current_epoch = epoch
                
                # Training phase
                train_loss, train_stats = self._train_epoch(train_loader)
                
                # Validation phase
                val_loss, val_dice = self._validate(val_loader)
                
                # Learning rate scheduling
                self.scheduler.step(val_dice)
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Log metrics
                self._log_epoch(train_loss, train_stats, val_loss, val_dice, current_lr, epoch)
                
                # Save best model
                is_best = val_dice > self.best_val_dice
                if is_best:
                    self.best_val_dice = val_dice
                
                self._save_checkpoint(
                    epoch,
                    {
                        'loss': val_loss,
                        'dice': val_dice
                    },
                    is_best=is_best
                )
                
                # Visualize predictions periodically
                if self.config.debug and epoch % 5 == 0:
                    self._visualize_predictions(val_loader, epoch)

        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving checkpoint...")
            self._save_checkpoint(
                self.current_epoch,
                {'dice': self.best_val_dice},
                is_best=False
            )
            print("Checkpoint saved. Exiting...")
            sys.exit(0)
            
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            raise
        
        self.writer.close()
        print("\nTraining completed!")
        print(f"Best validation Dice score: {self.best_val_dice:.4f}")

    def _train_epoch(self, train_loader):
        self.model.train()
        train_loss = 0
        valid_batches = 0
        total_norm = 0
        
        # Track statistics
        tumor_ratios = []
        mean_probs = []
        max_probs = []
        dice_scores = []
        
        with tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}/{self.config.num_epochs}") as pbar:
            for batch_idx, (images, tumor_masks, _) in enumerate(pbar):
                # Move data to device
                images = images.to(self.device)
                tumor_masks = tumor_masks.to(self.device)
                
                # Use CT image directly (no kidney channel)
                inputs = images[:, 0:1].float()
                
                # Forward pass
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    
                    # Ensure output shape matches target
                    if outputs.shape[2:] != tumor_masks.shape[2:]:
                        outputs = F.interpolate(
                            outputs,
                            size=tumor_masks.shape[2:],
                            mode='trilinear',
                            align_corners=False
                        )
                    
                    # Compute loss (passing ones for kidney_mask)
                    dummy_kidney = torch.ones_like(tumor_masks)
                    loss = self.criterion(outputs, tumor_masks, dummy_kidney)
                
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
                    stats = self.criterion.log_stats(outputs, tumor_masks, dummy_kidney)
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
        avg_stats = {
            'loss': train_loss / valid_batches if valid_batches > 0 else float('inf'),
            'tumor_ratio': np.mean(tumor_ratios),
            'mean_prob': np.mean(mean_probs),
            'max_prob': np.mean(max_probs),
            'dice_score': np.mean(dice_scores)
        }
        
        return avg_stats['loss'], avg_stats

    def _validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        val_dice = 0
        valid_batches = 0
        
        with tqdm(val_loader, desc="Validating") as pbar:
            for images, tumor_masks, _ in pbar:
                # Move to device
                images = images.to(self.device)
                tumor_masks = tumor_masks.to(self.device)
                
                # Use CT image only
                inputs = images[:, 0:1].float()
                
                try:
                    with torch.no_grad():
                        # Forward pass
                        outputs = self.model(inputs)
                        
                        # Ensure output shape matches target
                        if outputs.shape[2:] != tumor_masks.shape[2:]:
                            outputs = F.interpolate(
                                outputs,
                                size=tumor_masks.shape[2:],
                                mode='trilinear',
                                align_corners=False
                            )
                        
                        # Compute loss with dummy kidney mask
                        dummy_kidney = torch.ones_like(tumor_masks)
                        loss = self.criterion(outputs, tumor_masks, dummy_kidney)
                        
                        # Get predictions
                        probs = torch.sigmoid(outputs)
                        preds = (probs > 0.5).float()
                        
                        # Calculate Dice
                        dice = self._calculate_dice(preds, tumor_masks, dummy_kidney)
                        
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
        step = epoch * 1000 + batch_idx
        for key, value in stats.items():
            self.writer.add_scalar(f'Batch/{key}', value, step)

    def _log_epoch(self, train_loss, train_stats, val_loss, val_dice, lr, epoch):
        # Training metrics
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/val', val_loss, epoch)
        self.writer.add_scalar('Dice/val', val_dice, epoch)
        self.writer.add_scalar('LearningRate', lr, epoch)
        
        # Training statistics
        for key, value in train_stats.items():
            self.writer.add_scalar(f'Train/{key}', value, epoch)
        
        print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_stats['dice_score']:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")
        print(f"Learning Rate: {lr:.6f}")
        print(f"Mean tumor ratio: {train_stats['tumor_ratio']:.4%}")

    def _visualize_predictions(self, val_loader, epoch):
        self.model.eval()
        
        # Get first validation batch
        images, tumor_masks, _ = next(iter(val_loader))
        images = images.to(self.device)
        tumor_masks = tumor_masks.to(self.device)
        
        with torch.no_grad():
            # Get predictions
            inputs = images[:, 0:1].float()
            outputs = self.model(inputs)
            
            # Ensure output shape matches target
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
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Plot CT
            axes[0].imshow(images[0, 0, slice_idx].cpu(), cmap='gray')
            axes[0].set_title('CT')
            
            # Plot ground truth tumor
            axes[1].imshow(tumor_masks[0, 0, slice_idx].cpu(), cmap='gray')
            axes[1].set_title('Ground Truth')
            
            # Plot prediction
            axes[2].imshow(probs[0, 0, slice_idx].cpu(), cmap='gray')
            axes[2].set_title(f'Prediction (Epoch {epoch+1})')
            
            # Save figure
            self.writer.add_figure('Predictions', fig, epoch)
            plt.close(fig)

    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
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

    def _handle_interrupt(self, signum, frame):
        print("\nInterrupt received. Saving checkpoint before exiting...")
        self._save_checkpoint(
            self.current_epoch,
            {'dice': self.best_val_dice},
            is_best=False
        )
        print("Checkpoint saved. Exiting gracefully.")
        sys.exit(0)