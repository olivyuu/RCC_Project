import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import numpy as np
from tqdm import tqdm
import signal
import sys
from torch.utils.tensorboard import SummaryWriter

from model import nnUNetv2
from dataset_volume import KiTS23VolumeDataset
from losses import DC_and_CE_loss

class nnUNetVolumeTrainer:
    def __init__(self, config, debug_logger=None):
        self.config = config
        self.debug_logger = debug_logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create checkpoint directory
        self.config.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize model
        self.model = nnUNetv2(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            features=config.features
        ).to(self.device)
        
        # Load patch-trained weights if specified
        if config.transfer_learning and config.patch_weights_path:
            self._load_patch_weights()
        
        # Initialize training components
        self.criterion = DC_and_CE_loss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.initial_lr,
            weight_decay=config.weight_decay
        )
        self.scaler = GradScaler()
        self.writer = SummaryWriter(f"runs/{config.experiment_name}_volume_fold_{config.fold}")
        
        # Training state
        self.start_epoch = 0
        self.best_val_dice = float('-inf')
        self.patience_counter = 0
        self.current_epoch = 0
        self.frozen_layers = False
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
        
        if config.resume_training:
            self._load_checkpoint()
    
    def _load_patch_weights(self):
        """Load weights from patch-trained model."""
        print(f"Loading patch-trained weights from {self.config.patch_weights_path}")
        checkpoint = torch.load(self.config.patch_weights_path)
        
        # Load only model weights, not optimizer state
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        if self.config.freeze_layers:
            self._freeze_early_layers()
    
    def _freeze_early_layers(self):
        """Freeze early layers of the model for transfer learning."""
        print("Freezing early layers for transfer learning")
        for name, param in self.model.named_parameters():
            if 'encoder' in name:  # Freeze encoder layers
                param.requires_grad = False
        self.frozen_layers = True
    
    def _unfreeze_layers(self):
        """Unfreeze all layers for fine-tuning."""
        print("Unfreezing all layers")
        for param in self.model.parameters():
            param.requires_grad = True
        self.frozen_layers = False
    
    def train(self, dataset_path: str):
        # Create dataset
        dataset = KiTS23VolumeDataset(
            dataset_path, 
            self.config,
            train=True,
            preprocess=self.config.preprocess
        )
        
        # Split for validation
        val_size = int(len(dataset) * self.config.validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.seed)
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.vol_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=KiTS23VolumeDataset.collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.vol_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=KiTS23VolumeDataset.collate_fn
        )
        
        print(f"Training on {len(train_dataset)} volumes")
        print(f"Validating on {len(val_dataset)} volumes")
        
        try:
            for epoch in range(self.start_epoch, self.config.num_epochs):
                self.current_epoch = epoch
                
                # Unfreeze layers if needed
                if self.frozen_layers and epoch >= self.config.freeze_epochs:
                    self._unfreeze_layers()
                
                # Training phase
                self.model.train()
                train_loss = 0
                train_dice = 0
                self.optimizer.zero_grad()  # Zero gradients at start of epoch
                
                with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}") as pbar:
                    for batch_idx, (images, targets) in enumerate(pbar):
                        images = images.to(self.device)
                        targets = targets.to(self.device)
                        
                        # Forward pass with mixed precision
                        with autocast(enabled=self.config.use_amp):
                            outputs = self.model(images)
                            if isinstance(outputs, list):
                                loss = self.criterion(outputs[0], targets)
                                # Add deep supervision losses
                                for aux_output in outputs[1:]:
                                    loss += 0.4 * self.criterion(aux_output, targets)
                                main_output = outputs[0]
                            else:
                                loss = self.criterion(outputs, targets)
                                main_output = outputs
                            
                            # Scale loss for gradient accumulation
                            loss = loss / self.config.vol_gradient_accumulation_steps
                        
                        # Backward pass with gradient accumulation
                        self.scaler.scale(loss).backward()
                        
                        if (batch_idx + 1) % self.config.vol_gradient_accumulation_steps == 0:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.optimizer.zero_grad()
                        
                        # Update metrics
                        train_loss += loss.item() * self.config.vol_gradient_accumulation_steps
                        dice = self._calculate_dice(main_output, targets)
                        train_dice += dice
                        
                        pbar.set_postfix({
                            'loss': loss.item() * self.config.vol_gradient_accumulation_steps,
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
                
                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs!")
                    break
                
        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving checkpoint...")
            self._save_checkpoint(
                self.current_epoch,
                {'dice': self.best_val_dice},
                is_best=False
            )
            sys.exit(0)
        
        self.writer.close()
        print(f"Training completed! Best validation Dice: {self.best_val_dice:.4f}")
    
    def _validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        val_dice = 0
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validating"):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                with autocast(enabled=self.config.use_amp):
                    outputs = self.model(images)
                    main_output = outputs[0] if isinstance(outputs, list) else outputs
                    loss = self.criterion(main_output, targets)
                
                val_loss += loss.item()
                val_dice += self._calculate_dice(main_output, targets)
        
        return val_loss / len(val_loader), val_dice / len(val_loader)
    
    def _calculate_dice(self, outputs, targets):
        """Calculate Dice score."""
        preds = torch.argmax(outputs, dim=1)
        if len(targets.shape) == len(preds.shape) + 1:
            targets = targets.squeeze(1)
        
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        
        return (2. * intersection + 1e-5) / (union + 1e-5)
    
    def _save_checkpoint(self, epoch, metrics, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_dice': self.best_val_dice,
            'patience_counter': self.patience_counter,
            'metrics': metrics,
            'frozen_layers': self.frozen_layers
        }
        
        # Save latest checkpoint
        if self.config.save_latest:
            latest_path = self.config.checkpoint_dir / "volume_latest.pth"
            torch.save(checkpoint, latest_path)
        
        # Save best model
        if is_best:
            best_path = self.config.checkpoint_dir / f"volume_best_dice_{metrics['dice']:.4f}.pth"
            torch.save(checkpoint, best_path)
        
        # Save periodic checkpoint
        if (epoch + 1) % self.config.save_frequency == 0:
            path = self.config.checkpoint_dir / f"volume_checkpoint_epoch_{epoch+1}.pth"
            torch.save(checkpoint, path)
    
    def _load_checkpoint(self):
        checkpoint_path = self.config.checkpoint_dir / self.config.checkpoint_file
        if not checkpoint_path.exists():
            print(f"No checkpoint found at {checkpoint_path}")
            return
        
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_dice = checkpoint['best_val_dice']
        self.patience_counter = checkpoint['patience_counter']
        self.frozen_layers = checkpoint.get('frozen_layers', False)
        
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