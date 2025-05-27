import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
import signal
import sys
from tqdm import tqdm

from dataset import KiTS23Dataset
from losses import DC_and_BCE_loss
from .model import SegmentationModel

class SegmentationTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create checkpoint directory
        self.config.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize model
        self.model = SegmentationModel(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            features=config.features
        ).to(self.device)
        
        # Initialize training components
        self.criterion = DC_and_BCE_loss()  # Combined loss for segmentation
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.initial_lr,
            weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=config.lr_reduce_factor,
            patience=config.lr_schedule_patience,
            verbose=True
        )
        self.scaler = GradScaler()
        self.writer = SummaryWriter(f"{config.log_dir}/{config.experiment_name}_fold_{config.fold}")
        
        # Initialize training state
        self.start_epoch = 0
        self.best_val_dice = float('-inf')
        self.patience_counter = 0
        self.current_epoch = 0
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
        
        # Load checkpoint if resuming
        if config.resume_training:
            self._load_checkpoint()

    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_dice': self.best_val_dice,
            'patience_counter': self.patience_counter,
            'metrics': metrics
        }
        
        # Save latest checkpoint
        if self.config.save_latest:
            latest_path = self.config.checkpoint_dir / "latest.pth"
            torch.save(checkpoint, latest_path)
            print(f"Saved latest checkpoint: {latest_path}")
        
        # Save best model
        if is_best:
            best_path = self.config.checkpoint_dir / f"best_model_dice_{metrics['dice']:.4f}.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")
        
        # Save periodic checkpoint
        if (epoch + 1) % self.config.save_frequency == 0:
            path = self.config.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(checkpoint, path)
            print(f"Saved periodic checkpoint: {path}")

    def _load_checkpoint(self):
        checkpoint_path = self.config.checkpoint_dir / self.config.checkpoint_file
        if not checkpoint_path.exists():
            print(f"No checkpoint found at {checkpoint_path}")
            return
        
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_dice = checkpoint['best_val_dice']
        self.patience_counter = checkpoint['patience_counter']
        
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

    def train(self, dataset_path: str):
        # Create dataset
        dataset = KiTS23Dataset(dataset_path, self.config, preprocess=self.config.preprocess)
        
        # Split dataset
        val_size = int(len(dataset) * self.config.validation_split)
        train_size = len(dataset) - val_size
        generator = torch.Generator().manual_seed(self.config.seed)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        print(f"Training on {len(train_dataset)} samples")
        print(f"Validating on {len(val_dataset)} samples")
        
        try:
            for epoch in range(self.start_epoch, self.config.num_epochs):
                self.current_epoch = epoch
                
                # Training phase
                self.model.train()
                train_loss = 0
                train_dice = 0
                
                with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}") as pbar:
                    for batch_idx, (images, targets) in enumerate(pbar):
                        images, targets = images.to(self.device), targets.to(self.device)
                        
                        # Forward pass with mixed precision
                        with autocast(enabled=self.config.use_amp):
                            outputs = self.model(images)
                            if isinstance(outputs, list):
                                loss = self.criterion(outputs[0], targets)
                                
                                # Add deep supervision losses
                                ds_weight = 0.4
                                for aux_output in outputs[1:]:
                                    loss += ds_weight * self.criterion(aux_output, targets)
                                    
                                main_output = outputs[0]
                            else:
                                loss = self.criterion(outputs, targets)
                                main_output = outputs
                        
                        # Backward pass
                        self.optimizer.zero_grad()
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        
                        # Update metrics
                        train_loss += loss.item()
                        main_output = outputs[0] if isinstance(outputs, list) else outputs
                        dice = self._calculate_dice(main_output, targets)
                        train_dice += dice
                        
                        # Update progress bar
                        pbar.set_postfix({
                            'loss': loss.item(),
                            'dice': dice
                        })
                
                # Calculate epoch metrics
                train_loss /= len(train_loader)
                train_dice /= len(train_loader)
                
                # Validation phase
                val_loss, val_dice = self._validate(val_loader)
                
                # Update learning rate
                self.scheduler.step(val_dice)
                
                # Log metrics
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Dice/train', train_dice, epoch)
                self.writer.add_scalar('Dice/val', val_dice, epoch)
                self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
                
                # Print epoch results
                print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
                print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
                print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")
                
                # Save checkpoints
                is_best = val_dice > self.best_val_dice
                if is_best:
                    self.best_val_dice = val_dice
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                self._save_checkpoint(
                    epoch,
                    {'loss': val_loss, 'dice': val_dice},
                    is_best=is_best
                )
                
                # Early stopping
                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs!")
                    break
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving checkpoint...")
            self._save_checkpoint(
                self.current_epoch,
                {'dice': self.best_val_dice},
                is_best=False
            )
            print("Checkpoint saved. Exiting...")
            sys.exit(0)
        
        self.writer.close()
        print("\nTraining completed!")
        print(f"Best validation Dice score: {self.best_val_dice:.4f}")

    @torch.no_grad()
    def _validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        val_dice = 0
        
        print("\nStarting validation...")
        with tqdm(val_loader, desc="Validating") as pbar:
            for images, targets in pbar:
                images, targets = images.to(self.device), targets.to(self.device)
                
                with autocast(enabled=self.config.use_amp):
                    outputs = self.model(images)
                    main_output = outputs[0] if isinstance(outputs, list) else outputs
                    loss = self.criterion(main_output, targets)
                
                dice = self._calculate_dice(main_output, targets)
                val_loss += loss.item()
                val_dice += dice
                
                pbar.set_postfix({
                    'loss': loss.item(),
                    'dice': dice
                })
        
        return val_loss / len(val_loader), val_dice / len(val_loader)

    def _calculate_dice(self, outputs, targets):
        if outputs.shape[-3:] != targets.shape[-3:]:
            outputs = F.interpolate(
                outputs,
                size=targets.shape[-3:],
                mode='trilinear',
                align_corners=False
            )
        
        # Convert logits to predictions
        preds = torch.argmax(outputs, dim=1)
        if len(targets.shape) == len(preds.shape) + 1:
            targets = targets.squeeze(1)
        
        # Calculate Dice coefficient for all classes
        dice_scores = []
        for class_idx in range(1, outputs.shape[1]):  # Skip background class
            class_pred = (preds == class_idx)
            class_target = (targets == class_idx)
            
            intersection = (class_pred & class_target).sum().float()
            union = class_pred.sum().float() + class_target.sum().float()
            
            dice = (2. * intersection + 1e-5) / (union + 1e-5)
            dice_scores.append(dice)
        
        return sum(dice_scores) / len(dice_scores)