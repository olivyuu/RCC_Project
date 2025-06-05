import multiprocessing as mp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler
import torch.cuda
from pathlib import Path
import numpy as np
from tqdm import tqdm
import signal
import sys
from torch.utils.tensorboard import SummaryWriter

from models.segmentation.model import SegmentationModel
from dataset_patch import KiTS23PatchDataset
from losses_patch import WeightedDiceBCELoss

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
            out_channels=1,  # Single channel for tumor probability
            features=config.features
        ).to(self.device)
        
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
        
        # Tighter LR scheduling
        self.scheduler = torch.optim.ReduceLROnPlateau(
            self.optimizer, 
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )
        
        self.scaler = GradScaler()
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        # Training state
        self.start_epoch = 0
        self.best_val_dice = float('-inf')
        self.current_epoch = 0
        self.max_grad_norm = 5.0
        
        # Patch training settings
        self.patch_size = (64, 128, 128)
        self.batch_size = 2  # Can be higher since patches are smaller
        self.num_workers = 4
        
        if config.resume_training:
            self._load_checkpoint()

    def train(self, dataset_path: str):
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass

        # Create training dataset
        from dataset_volume import KiTS23VolumeDataset
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
            tumor_only_prob=0.7,  # 70% tumor-centered patches
            debug=False
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
        
        # Keep validation on full volumes
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=KiTS23VolumeDataset.collate_fn
        )

        print(f"\nTraining configuration:")
        print(f"Patch size: {self.patch_size}")
        print(f"Batch size: {self.batch_size}")
        print(f"Training patches per epoch: {len(train_loader)}")
        print(f"Validation volumes: {len(val_dataset)}")

        try:
            for epoch in range(self.start_epoch, self.config.num_epochs):
                self.current_epoch = epoch
                
                # Training phase
                self.model.train()
                train_loss = 0
                train_dice = 0
                valid_batches = 0
                total_norm = 0
                
                # Track tumor ratio in patches
                tumor_ratios = []
                
                with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}") as pbar:
                    for batch_idx, (images, tumor_masks, kidney_masks) in enumerate(pbar):
                        # Move data to device
                        images = images.to(device=self.device)
                        tumor_masks = tumor_masks.to(device=self.device)
                        kidney_masks = kidney_masks.to(device=self.device)
                        
                        # Get CT and kidney channels
                        ct_images = images[:, 0:1].float()
                        kidney_inputs = kidney_masks.float()
                        
                        # Combine input channels
                        inputs = torch.cat([ct_images, kidney_inputs], dim=1)
                        
                        # Forward pass
                        with torch.cuda.amp.autocast():
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, tumor_masks, kidney_masks)
                        
                        # Backward pass with gradient clipping
                        self.optimizer.zero_grad()
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        
                        # Clip gradients
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.max_grad_norm
                        )
                        
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        
                        # Compute metrics
                        with torch.no_grad():
                            # Get statistics
                            stats = self.criterion.log_stats(outputs, tumor_masks, kidney_masks)
                            tumor_ratios.append(stats['tumor_ratio'])
                            
                            if batch_idx % 50 == 0:
                                self._log_batch_stats(stats, batch_idx, epoch)
                            
                            train_loss += loss.item()
                            valid_batches += 1
                        
                        # Update progress bar
                        pbar.set_postfix({
                            'loss': f"{loss.item():.4f}",
                            'tumor_ratio': f"{stats['tumor_ratio']:.4%}",
                            'grad_norm': f"{total_norm:.4f}"
                        })
                        
                        # Clear cache periodically
                        if batch_idx % 10 == 0:
                            torch.cuda.empty_cache()
                
                # Calculate epoch metrics
                train_loss = train_loss / valid_batches if valid_batches > 0 else float('inf')
                mean_tumor_ratio = np.mean(tumor_ratios)
                
                print(f"\nTraining epoch statistics:")
                print(f"Mean tumor ratio in patches: {mean_tumor_ratio:.4%}")
                print(f"Mean loss: {train_loss:.4f}")
                
                # Validation phase
                val_loss, val_dice = self._validate(val_loader)
                
                # Learning rate scheduling
                self.scheduler.step(val_dice)
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Log metrics
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Dice/val', val_dice, epoch)
                self.writer.add_scalar('LearningRate', current_lr, epoch)
                self.writer.add_scalar('TumorRatio', mean_tumor_ratio, epoch)
                
                print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
                print(f"Train - Loss: {train_loss:.4f}")
                print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")
                print(f"Learning Rate: {current_lr:.6f}")
                
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

    def _log_batch_stats(self, stats: dict, batch_idx: int, epoch: int):
        """Log detailed batch statistics to tensorboard"""
        step = epoch * 1000 + batch_idx  # Unique step number
        self.writer.add_scalar('Batch/tumor_ratio', stats['tumor_ratio'], step)
        self.writer.add_scalar('Batch/true_positives', stats['true_positives'], step)
        self.writer.add_scalar('Batch/mean_prob', stats['mean_prob'], step)
        self.writer.add_scalar('Batch/max_prob', stats['max_prob'], step)

    @torch.no_grad()
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
                    # Forward pass
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, tumor_masks, kidney_masks)
                    
                    # Get predictions
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
        
        if valid_batches > 0:
            return (val_loss / valid_batches,
                   val_dice / valid_batches)
        return float('inf'), 0

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

    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save training checkpoint"""
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
        
        print(f"Resuming training from epoch {self.start_epoch}")