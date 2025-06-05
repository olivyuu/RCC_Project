import multiprocessing as mp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
import torch.cuda
from pathlib import Path
import numpy as np
from tqdm import tqdm
import signal
import sys
from torch.utils.tensorboard import SummaryWriter

from models.segmentation.model import SegmentationModel
from dataset_volume import KiTS23VolumeDataset
from losses import DC_and_BCE_loss

class SegmentationTrainer:
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
            in_channels=2,  # Image and kidney mask channels
            out_channels=2,  # Two channels for background + tumor
            features=config.features
        ).to(self.device)
        
        self.model.enable_checkpointing()
        self._initialize_weights()
        
        # Initialize training components
        self.criterion = DC_and_BCE_loss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-5,
            eps=1e-8
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
        
        self.scaler = GradScaler()
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        # Training state
        self.start_epoch = 0
        self.best_val_dice = float('-inf')
        self.current_epoch = 0
        self.max_grad_norm = 1.0
        self.accum_steps = getattr(config, 'vol_gradient_accumulation_steps', 4)
        
        if config.resume_training:
            self._load_checkpoint()

    def _monitor_predictions(self, outputs, targets):
        """Monitor prediction statistics"""
        with torch.no_grad():
            # Get model statistics
            stats = self.model.get_output_stats(outputs)
            
            # Compute soft Dice
            soft_dice_stats = self.model.compute_soft_dice(outputs, targets)
            
            # Combine all stats
            stats.update(soft_dice_stats)
            
            # Log detailed statistics
            print("\nPrediction Statistics:")
            print(f"  Soft Dice: {stats['soft_dice']:.4f}")
            print(f"  Average tumor probability: {stats['avg_tumor_prob']:.4f}")
            print(f"  Tumor probability range: [{stats['tumor_prob_min']:.4f}, {stats['tumor_prob_max']:.4f}]")
            print(f"  Voxels > 0.5: {stats['tumor_voxels_gt_50']}")
            print(f"  Voxels > 0.1: {stats['tumor_voxels_gt_10']}")
            
            return stats

    def train(self, dataset_path: str):
        # Set multiprocessing method to spawn
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass

        dataset = KiTS23VolumeDataset(dataset_path, self.config, preprocess=self.config.preprocess)
        
        val_size = int(len(dataset) * 0.2)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=getattr(self.config, 'vol_batch_size', 1),
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=KiTS23VolumeDataset.collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=KiTS23VolumeDataset.collate_fn
        )
        
        print(f"\nTraining on {len(train_dataset)} volumes")
        print(f"Validating on {len(val_dataset)} volumes")
        print(f"Using gradient accumulation steps: {self.accum_steps}")
        
        try:
            for epoch in range(self.start_epoch, self.config.num_epochs):
                self.current_epoch = epoch
                
                # Training phase
                self.model.train()
                train_loss = 0
                train_dice = 0
                train_soft_dice = 0
                valid_batches = 0
                total_norm = 0
                
                self.optimizer.zero_grad()
                
                with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}") as pbar:
                    for batch_idx, (images, targets) in enumerate(pbar):
                        images = images.to(dtype=torch.float32, device=self.device)
                        targets = targets.to(dtype=torch.float32, device=self.device)
                        
                        if torch.isnan(images).any() or torch.isinf(images).any():
                            print(f"\nWarning: Found NaN/Inf in input images at epoch {epoch}, batch {batch_idx}")
                            continue
                            
                        if torch.isnan(targets).any() or torch.isinf(targets).any():
                            print(f"\nWarning: Found NaN/Inf in target masks at epoch {epoch}, batch {batch_idx}")
                            continue
                        
                        try:
                            with autocast():
                                outputs = self.model(images)
                                if isinstance(outputs, (list, tuple)):
                                    outputs = outputs[-1]
                                
                                if outputs.shape[-3:] != targets.shape[-3:]:
                                    outputs = F.interpolate(
                                        outputs,
                                        size=targets.shape[-3:],
                                        mode='trilinear',
                                        align_corners=False
                                    )
                                
                                loss = self.criterion(outputs, targets) / self.accum_steps
                            
                            if torch.isnan(loss) or torch.isinf(loss):
                                print("\nWarning: Invalid loss value detected!")
                                print(f"Loss: {loss.item()}")
                                continue
                            
                            # Monitor predictions periodically
                            if batch_idx % 50 == 0:
                                stats = self._monitor_predictions(outputs, targets)
                                self.writer.add_scalar('Training/SoftDice', stats['soft_dice'], epoch * len(train_loader) + batch_idx)
                                self.writer.add_scalar('Training/AvgTumorProb', stats['avg_tumor_prob'], epoch * len(train_loader) + batch_idx)
                            
                            self.scaler.scale(loss).backward()
                            
                            if (batch_idx + 1) % self.accum_steps == 0:
                                self.scaler.unscale_(self.optimizer)
                                total_norm = torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(),
                                    self.max_grad_norm
                                )
                                
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                                self.optimizer.zero_grad()
                                
                                torch.cuda.empty_cache()
                            
                            with torch.no_grad():
                                probs = torch.softmax(outputs, dim=1)[:, 1:]
                                hard_dice = self._calculate_dice(probs > 0.5, targets)
                                stats = self.model.compute_soft_dice(outputs, targets)
                                soft_dice = stats['soft_dice']
                                
                                if not torch.isnan(hard_dice) and not torch.isinf(hard_dice):
                                    train_loss += loss.item() * self.accum_steps
                                    train_dice += hard_dice
                                    train_soft_dice += soft_dice
                                    valid_batches += 1
                            
                            if batch_idx % 10 == 0:
                                torch.cuda.empty_cache()
                                
                                if total_norm > 0:
                                    print(f"\nGradient norm: {total_norm:.4f}")
                                print(f"Loss: {loss.item() * self.accum_steps:.4f}")
                                print(f"Hard Dice: {hard_dice:.4f}")
                                print(f"Soft Dice: {soft_dice:.4f}")
                                
                                if torch.cuda.is_available():
                                    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                                    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
                            
                            pbar.set_postfix({
                                'loss': f"{loss.item() * self.accum_steps:.4f}",
                                'hard_dice': f"{hard_dice:.4f}",
                                'soft_dice': f"{soft_dice:.4f}",
                                'grad_norm': f"{total_norm:.4f}" if total_norm > 0 else "N/A"
                            })
                            
                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                print(f"\nCUDA OOM in batch {batch_idx}. Attempting recovery...")
                                if hasattr(torch.cuda, 'empty_cache'):
                                    torch.cuda.empty_cache()
                                continue
                            else:
                                raise e
                
                # Calculate epoch metrics
                if valid_batches > 0:
                    train_loss /= valid_batches
                    train_dice /= valid_batches
                    train_soft_dice /= valid_batches
                else:
                    print("Warning: No valid batches in epoch!")
                    continue
                
                torch.cuda.empty_cache()
                
                # Validation phase
                val_loss, val_dice, val_soft_dice = self._validate(val_loader)
                
                torch.cuda.empty_cache()
                
                # Learning rate scheduling
                self.scheduler.step(val_soft_dice)  # Use soft Dice for scheduling
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Log metrics
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Dice/train_hard', train_dice, epoch)
                self.writer.add_scalar('Dice/val_hard', val_dice, epoch)
                self.writer.add_scalar('Dice/train_soft', train_soft_dice, epoch)
                self.writer.add_scalar('Dice/val_soft', val_soft_dice, epoch)
                self.writer.add_scalar('LearningRate', current_lr, epoch)
                
                print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
                print(f"Train - Loss: {train_loss:.4f}, Hard Dice: {train_dice:.4f}, Soft Dice: {train_soft_dice:.4f}")
                print(f"Val   - Loss: {val_loss:.4f}, Hard Dice: {val_dice:.4f}, Soft Dice: {val_soft_dice:.4f}")
                print(f"Learning Rate: {current_lr:.6f}")
                
                # Save best model based on soft Dice
                is_best = val_soft_dice > self.best_val_dice
                if is_best:
                    self.best_val_dice = val_soft_dice
                
                self._save_checkpoint(
                    epoch,
                    {
                        'loss': val_loss,
                        'hard_dice': val_dice,
                        'soft_dice': val_soft_dice
                    },
                    is_best=is_best
                )
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving checkpoint...")
            self._save_checkpoint(
                self.current_epoch,
                {'soft_dice': self.best_val_dice},
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
        print(f"Best validation soft Dice score: {self.best_val_dice:.4f}")

    @torch.no_grad()
    def _validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        val_dice = 0
        val_soft_dice = 0
        valid_batches = 0
        
        with tqdm(val_loader, desc="Validating") as pbar:
            for batch_idx, (images, targets) in enumerate(pbar):
                images = images.to(dtype=torch.float32, device=self.device)
                targets = targets.to(dtype=torch.float32, device=self.device)
                
                if torch.isnan(images).any() or torch.isinf(images).any():
                    print(f"\nWarning: Found NaN/Inf in validation images, batch {batch_idx}")
                    continue
                    
                if torch.isnan(targets).any() or torch.isinf(targets).any():
                    print(f"\nWarning: Found NaN/Inf in validation targets, batch {batch_idx}")
                    continue
                
                try:
                    with autocast():
                        outputs = self.model(images)
                        if isinstance(outputs, (list, tuple)):
                            outputs = outputs[-1]
                        
                        if outputs.shape[-3:] != targets.shape[-3:]:
                            outputs = F.interpolate(
                                outputs,
                                size=targets.shape[-3:],
                                mode='trilinear',
                                align_corners=False
                            )
                        
                        loss = self.criterion(outputs, targets)
                        
                    # Monitor predictions periodically during validation
                    if batch_idx % 10 == 0:
                        self._monitor_predictions(outputs, targets)
                    
                    probs = torch.softmax(outputs, dim=1)[:, 1:]
                    hard_dice = self._calculate_dice(probs > 0.5, targets)
                    stats = self.model.compute_soft_dice(outputs, targets)
                    soft_dice = stats['soft_dice']
                    
                    if not torch.isnan(hard_dice) and not torch.isinf(hard_dice):
                        val_loss += loss.item()
                        val_dice += hard_dice
                        val_soft_dice += soft_dice
                        valid_batches += 1
                    
                    if batch_idx % 5 == 0:
                        torch.cuda.empty_cache()
                    
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'hard_dice': f"{hard_dice:.4f}",
                        'soft_dice': f"{soft_dice:.4f}"
                    })
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\nCUDA OOM in validation batch {batch_idx}. Attempting recovery...")
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        if valid_batches > 0:
            return (val_loss / valid_batches, 
                   val_dice / valid_batches,
                   val_soft_dice / valid_batches)
        else:
            return float('inf'), 0, 0

    def _calculate_dice(self, outputs, targets, smooth=1e-5):
        """Calculate hard Dice score using thresholded predictions"""
        intersection = (outputs * targets).sum()
        union = outputs.sum() + targets.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        
        if dice < 0.1:
            print("\nLow dice score detected!")
            print(f"Predictions sum: {outputs.sum().item()}")
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
        
        latest_path = self.config.checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)
        
        if is_best:
            best_path = self.config.checkpoint_dir / f"best_model_dice_{metrics['soft_dice']:.4f}.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model with soft Dice: {metrics['soft_dice']:.4f}")

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
            {'soft_dice': self.best_val_dice},
            is_best=False
        )
        print("Checkpoint saved. Exiting gracefully.")
        sys.exit(0)