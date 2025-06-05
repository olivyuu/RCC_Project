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

from models.segmentation.model import SegmentationModel  # Updated import
from dataset_volume import KiTS23VolumeDataset
from losses import DC_and_BCE_loss

class SegmentationTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Configure CUDA memory allocator
        if torch.cuda.is_available():
            # Set memory allocator for better fragmentation handling
            torch.cuda.set_per_process_memory_fraction(0.95)  # Reserve some memory for system
            torch.cuda.empty_cache()
            # Set allocation config for better memory management
            torch.cuda.set_allocator_settings(
                max_split_size_mb=128,  # Reduce fragmentation
                garbage_collection_threshold=0.8  # More aggressive GC
            )
            # Set deterministic mode for better memory behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Create checkpoint directory
        self.config.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize model
        self.model = SegmentationModel(
            in_channels=2,  # Image and kidney mask channels
            out_channels=1,  # Tumor segmentation
            features=config.features
        ).to(self.device)
        
        # Enable gradient checkpointing for memory efficiency
        self.model.enable_checkpointing()
        
        # Initialize weights with He initialization
        self._initialize_weights()
        
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
        
        self.scaler = GradScaler()  # For mixed precision training
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        # Training state
        self.start_epoch = 0
        self.best_val_dice = float('-inf')
        self.current_epoch = 0
        self.max_grad_norm = 1.0
        
        if config.resume_training:
            self._load_checkpoint()

    def _initialize_weights(self):
        """Initialize network weights using He initialization"""
        for m in self.model.modules():
            if isinstance(m, (torch.nn.Conv3d, torch.nn.ConvTranspose3d)):
                torch.nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.BatchNorm3d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.zeros_(m.bias)

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
                        
                        # Clear gradients
                        self.optimizer.zero_grad()
                        
                        # Forward pass with mixed precision
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
                            
                            # Calculate loss
                            loss = self.criterion(outputs, targets)
                        
                        if torch.isnan(loss) or torch.isinf(loss):
                            print("\nWarning: Invalid loss value detected!")
                            print(f"Loss: {loss.item()}")
                            continue
                        
                        # Backward pass with gradient scaling
                        self.scaler.scale(loss).backward()
                        
                        # Unscale before gradient clipping
                        self.scaler.unscale_(self.optimizer)
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.max_grad_norm
                        )
                        
                        # Optimizer step with scaling
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        
                        # Calculate metrics
                        with torch.no_grad():
                            outputs_sigmoid = torch.sigmoid(outputs)
                            dice = self._calculate_dice(outputs_sigmoid.detach(), targets)
                            
                            if not torch.isnan(dice) and not torch.isinf(dice):
                                train_loss += loss.item()
                                train_dice += dice
                                valid_batches += 1
                        
                        # Clear GPU cache periodically
                        if batch_idx % 10 == 0:
                            torch.cuda.empty_cache()
                            
                            # Logging
                            print(f"\nGradient norm: {total_norm:.4f}")
                            print(f"Loss: {loss.item():.4f}")
                            print(f"Dice: {dice.item():.4f}")
                            
                            # Log GPU memory
                            if torch.cuda.is_available():
                                print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                                print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
                        
                        pbar.set_postfix({
                            'loss': f"{loss.item():.4f}",
                            'dice': f"{dice.item():.4f}",
                            'grad_norm': f"{total_norm:.4f}"
                        })
                
                # Calculate epoch metrics
                if valid_batches > 0:
                    train_loss /= valid_batches
                    train_dice /= valid_batches
                else:
                    print("Warning: No valid batches in epoch!")
                    continue
                
                # Clear cache before validation
                torch.cuda.empty_cache()
                
                # Validation phase
                val_loss, val_dice = self._validate(val_loader)
                
                # Clear cache after validation
                torch.cuda.empty_cache()
                
                # Learning rate scheduling
                self.scheduler.step(val_dice)
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Log metrics
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Dice/train', train_dice, epoch)
                self.writer.add_scalar('Dice/val', val_dice, epoch)
                self.writer.add_scalar('LearningRate', current_lr, epoch)
                
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
            for batch_idx, (images, targets) in enumerate(pbar):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass with mixed precision
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
                    outputs_sigmoid = torch.sigmoid(outputs)
                    dice = self._calculate_dice(outputs_sigmoid, targets)
                
                if not torch.isnan(dice) and not torch.isinf(dice):
                    val_loss += loss.item()
                    val_dice += dice
                    valid_batches += 1
                
                # Clear GPU cache periodically
                if batch_idx % 5 == 0:
                    torch.cuda.empty_cache()
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'dice': f"{dice.item():.4f}"
                })
        
        if valid_batches > 0:
            return val_loss / valid_batches, val_dice / valid_batches
        else:
            return float('inf'), 0

    def _calculate_dice(self, outputs, targets, smooth=1e-5):
        preds = (outputs > 0.5).float()
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        
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
        
        latest_path = self.config.checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)
        
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