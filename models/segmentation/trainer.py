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

from model import nnUNetv2
from dataset_volume import KiTS23VolumeDataset
from losses import DC_and_BCE_loss

class SegmentationTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
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
            lr=config.initial_lr,
            weight_decay=0.0001
        )
        self.scaler = GradScaler()
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        # Initialize training state
        self.start_epoch = 0
        self.best_val_dice = float('-inf')
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

    def train(self, dataset_path: str):
        # Create dataset
        dataset = KiTS23VolumeDataset(dataset_path, self.config, preprocess=self.config.preprocess)
        
        # Split dataset
        val_size = int(len(dataset) * 0.2)  # 20% validation split
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create dataloaders
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
            batch_size=1,  # Use batch size 1 for validation
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
                
                with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}") as pbar:
                    for batch_idx, (images, targets) in enumerate(pbar):
                        images, targets = images.to(self.device), targets.to(self.device)
                        
                        # Forward pass with mixed precision
                        with autocast():
                            outputs = self.model(images)
                            if isinstance(outputs, tuple):
                                outputs = outputs[0]  # Take main output if model returns multiple outputs
                            loss = self.criterion(outputs, targets)
                        
                        # Backward pass
                        self.optimizer.zero_grad()
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        
                        # Calculate metrics
                        dice = self._calculate_dice(outputs.detach(), targets)
                        train_loss += loss.item()
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
                
                # Log metrics
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Dice/train', train_dice, epoch)
                self.writer.add_scalar('Dice/val', val_dice, epoch)
                
                # Print epoch results
                print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
                print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
                print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")
                
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
        
        self.writer.close()
        print("\nTraining completed!")
        print(f"Best validation Dice score: {self.best_val_dice:.4f}")

    @torch.no_grad()
    def _validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        val_dice = 0
        
        with tqdm(val_loader, desc="Validating") as pbar:
            for images, targets in pbar:
                images, targets = images.to(self.device), targets.to(self.device)
                
                with autocast():
                    outputs = self.model(images)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = self.criterion(outputs, targets)
                
                dice = self._calculate_dice(outputs, targets)
                val_loss += loss.item()
                val_dice += dice
                
                pbar.set_postfix({
                    'loss': loss.item(),
                    'dice': dice
                })
        
        return val_loss / len(val_loader), val_dice / len(val_loader)

    def _calculate_dice(self, outputs, targets):
        # Convert logits to binary predictions
        preds = (torch.sigmoid(outputs) > 0.5).float()
        
        # Calculate dice coefficient
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        
        return (2. * intersection + 1e-5) / (union + 1e-5)