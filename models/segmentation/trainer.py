import multiprocessing as mp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import numpy as np
from tqdm import tqdm
import signal
import sys
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from model import nnUNetv2
from dataset_volume import KiTS23VolumeDataset
from losses import DC_and_BCE_loss

class DebugStats:
    """Helper class to track and log training statistics"""
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir) / "debug_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {
            'gradient_norms': [],
            'weight_norms': [],
            'activation_ranges': [],
            'loss_values': [],
            'dice_scores': []
        }
    
    def log_batch(self, epoch, batch_idx, model, loss, dice, outputs, targets):
        # Log gradients and weights
        grad_norm = 0
        weight_norm = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item()
            weight_norm += param.norm().item()
        
        self.stats['gradient_norms'].append(grad_norm)
        self.stats['weight_norms'].append(weight_norm)
        
        # Log outputs range
        output_min = outputs.min().item()
        output_max = outputs.max().item()
        self.stats['activation_ranges'].append((output_min, output_max))
        
        # Log metrics
        self.stats['loss_values'].append(loss.item())
        self.stats['dice_scores'].append(dice.item())
        
        # Save detailed batch info periodically
        if batch_idx % 10 == 0:
            self._save_batch_info(epoch, batch_idx, model, outputs, targets)
    
    def _save_batch_info(self, epoch, batch_idx, model, outputs, targets):
        info_file = self.log_dir / f"batch_e{epoch}_b{batch_idx}.txt"
        with open(info_file, "w") as f:
            f.write(f"=== Batch Information ===\n")
            f.write(f"Epoch: {epoch}, Batch: {batch_idx}\n\n")
            
            f.write("Recent Statistics:\n")
            f.write(f"Last 5 gradient norms: {self.stats['gradient_norms'][-5:]}\n")
            f.write(f"Last 5 weight norms: {self.stats['weight_norms'][-5:]}\n")
            f.write(f"Last 5 loss values: {self.stats['loss_values'][-5:]}\n")
            f.write(f"Last 5 dice scores: {self.stats['dice_scores'][-5:]}\n\n")
            
            f.write("Layer Information:\n")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    f.write(f"\n{name}:\n")
                    f.write(f"Weight range: [{param.min().item():.4f}, {param.max().item():.4f}]\n")
                    f.write(f"Gradient range: [{param.grad.min().item():.4f}, {param.grad.max().item():.4f}]\n")
                    f.write(f"Weight norm: {param.norm().item():.4f}\n")
                    f.write(f"Gradient norm: {param.grad.norm().item():.4f}\n")
            
            f.write("\nPrediction Statistics:\n")
            f.write(f"Output range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]\n")
            f.write(f"Output mean: {outputs.mean().item():.4f}\n")
            f.write(f"Output std: {outputs.std().item():.4f}\n")
            f.write(f"Target range: [{targets.min().item():.4f}, {targets.max().item():.4f}]\n")
            f.write(f"Target mean: {targets.mean().item():.4f}\n")
            f.write(f"Positive target ratio: {(targets > 0).float().mean().item():.4f}\n")

class SegmentationTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize debug stats using log_dir
        self.debug_stats = DebugStats(config.log_dir)
        
        # Create checkpoint directory
        self.config.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize model
        self.model = nnUNetv2(
            in_channels=2,  # Image and kidney mask channels
            out_channels=1,  # Tumor segmentation
            features=config.features
        ).to(self.device)
        
        # Initialize training components
        self.criterion = DC_and_BCE_loss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-4,  # Lower learning rate
            weight_decay=1e-5,  # L2 regularization
            eps=1e-8  # For numerical stability
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
        
        self.scaler = GradScaler()
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        # Initialize training state
        self.start_epoch = 0
        self.best_val_dice = float('-inf')
        self.current_epoch = 0
        self.max_grad_norm = 1.0
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
        
        if config.resume_training:
            self._load_checkpoint()

    def train(self, dataset_path: str):
        dataset = KiTS23VolumeDataset(dataset_path, self.config, preprocess=self.config.preprocess)
        
        val_size = int(len(dataset) * 0.2)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=KiTS23VolumeDataset.collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=KiTS23VolumeDataset.collate_fn
        )
        
        print(f"\nTraining on {len(train_dataset)} volumes")
        print(f"Validating on {len(val_dataset)} volumes")
        
        try:
            for epoch in range(self.start_epoch, self.config.num_epochs):
                self.current_epoch = epoch
                
                # Training phase
                self.model.train()
                train_loss = 0
                train_dice = 0
                valid_batches = 0
                
                with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}") as pbar:
                    for batch_idx, (images, targets) in enumerate(pbar):
                        images = images.to(self.device)
                        targets = targets.to(self.device)
                        
                        # Forward pass with mixed precision
                        with autocast():
                            outputs = self.model(images)
                            if isinstance(outputs, (list, tuple)):
                                outputs = outputs[-1]
                            
                            # Debug pre-interpolation
                            if batch_idx % 10 == 0:
                                print(f"\nPre-interpolation shape: {outputs.shape}")
                                print(f"Pre-interpolation range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
                            
                            if outputs.shape[-3:] != targets.shape[-3:]:
                                outputs = F.interpolate(
                                    outputs.float(),
                                    size=targets.shape[-3:],
                                    mode='trilinear',
                                    align_corners=False
                                )
                                
                                # Debug post-interpolation
                                if batch_idx % 10 == 0:
                                    print(f"Post-interpolation shape: {outputs.shape}")
                                    print(f"Post-interpolation range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
                            
                            # Apply sigmoid for binary classification
                            outputs = torch.sigmoid(outputs)
                            
                            # Debug final outputs
                            if batch_idx % 10 == 0:
                                print(f"Final output range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
                                print(f"Positive prediction ratio: {(outputs > 0.5).float().mean().item():.4f}")
                                print(f"Target positive ratio: {(targets > 0).float().mean().item():.4f}")
                            
                            loss = self.criterion(outputs, targets)
                        
                        # Backward pass with gradient clipping
                        self.optimizer.zero_grad()
                        self.scaler.scale(loss).backward()
                        
                        # Log gradients before clipping
                        if batch_idx % 10 == 0:
                            grad_norms = {name: param.grad.norm().item() 
                                        for name, param in self.model.named_parameters() 
                                        if param.grad is not None}
                            print("\nGradient norms before clipping:")
                            for name, norm in grad_norms.items():
                                print(f"{name}: {norm:.4f}")
                        
                        self.scaler.unscale_(self.optimizer)
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.max_grad_norm
                        )
                        
                        if batch_idx % 10 == 0:
                            print(f"Total gradient norm after clipping: {total_norm:.4f}")
                        
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        
                        # Calculate metrics
                        with torch.no_grad():
                            dice = self._calculate_dice(outputs.detach(), targets)
                            if not torch.isnan(dice) and not torch.isinf(dice):
                                train_loss += loss.item()
                                train_dice += dice
                                valid_batches += 1
                                
                                # Log batch statistics
                                self.debug_stats.log_batch(
                                    epoch, batch_idx, self.model,
                                    loss, dice, outputs, targets
                                )
                        
                        # Update progress bar
                        pbar.set_postfix({
                            'loss': f"{loss.item():.4f}",
                            'dice': f"{dice.item():.4f}",
                            'grad_norm': f"{total_norm:.4f}",
                            'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                        })
                
                # Calculate epoch metrics
                if valid_batches > 0:
                    train_loss /= valid_batches
                    train_dice /= valid_batches
                else:
                    print("Warning: No valid batches in epoch!")
                    continue
                
                # Validation phase
                val_loss, val_dice = self._validate(val_loader)
                
                # Learning rate scheduling
                self.scheduler.step(val_dice)
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Log metrics
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Dice/train', train_dice, epoch)
                self.writer.add_scalar('Dice/val', val_dice, epoch)
                self.writer.add_scalar('LearningRate', current_lr, epoch)
                
                # Save histograms
                for name, param in self.model.named_parameters():
                    self.writer.add_histogram(f'weights/{name}', param, epoch)
                    if param.grad is not None:
                        self.writer.add_histogram(f'grads/{name}', param.grad, epoch)
                
                # Print epoch results
                print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
                print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
                print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")
                print(f"Learning Rate: {current_lr:.6f}")
                
                # Save best model
                is_best = val_dice > self.best_val_dice
                if is_best:
                    self.best_val_dice = val_dice
                
                self._save_checkpoint(
                    epoch,
                    {'loss': val_loss, 'dice': val_dice},
                    is_best=is_best
                )
                
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

    @torch.no_grad()
    def _validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        val_dice = 0
        valid_batches = 0
        
        with tqdm(val_loader, desc="Validating") as pbar:
            for images, targets in pbar:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                with autocast():
                    outputs = self.model(images)
                    if isinstance(outputs, (list, tuple)):
                        outputs = outputs[-1]
                    
                    if outputs.shape[-3:] != targets.shape[-3:]:
                        outputs = F.interpolate(
                            outputs.float(),
                            size=targets.shape[-3:],
                            mode='trilinear',
                            align_corners=False
                        )
                    
                    outputs = torch.sigmoid(outputs)
                    loss = self.criterion(outputs, targets)
                
                dice = self._calculate_dice(outputs, targets)
                if not torch.isnan(dice) and not torch.isinf(dice):
                    val_loss += loss.item()
                    val_dice += dice
                    valid_batches += 1
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'dice': f"{dice.item():.4f}"
                })
        
        if valid_batches > 0:
            return val_loss / valid_batches, val_dice / valid_batches
        else:
            return float('inf'), 0

    def _calculate_dice(self, outputs, targets, smooth=1e-5):
        # Already have sigmoid in forward pass
        preds = (outputs > 0.5).float()
        
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        
        # Debug low dice scores
        if dice < 0.1:
            print("\nLow dice score detected!")
            print(f"Predictions sum: {preds.sum().item()}")
            print(f"Targets sum: {targets.sum().item()}")
            print(f"Intersection: {intersection.item()}")
            print(f"Union: {union.item()}")
        
        return dice

    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_dice': self.best_val_dice,
            'metrics': metrics
        }
        
        # Save latest checkpoint
        latest_path = self.config.checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)
        
        # Save best model
        if is_best:
            best_path = self.config.checkpoint_dir / f"best_model_dice_{metrics['dice']:.4f}.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")

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