import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import numpy as np
import random  # Added for seeding
from tqdm import tqdm
import signal
import sys
from torch.utils.tensorboard import SummaryWriter

from model import nnUNetv2
from dataset import KiTS23Dataset
from losses import DC_and_CE_loss
from config import nnUNetConfig


# Function for setting random seeds
def set_seed(seed: int, deterministic: bool = False, benchmark: bool = True):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    
    # Handle cuDNN settings
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    print(f"Set seed to {seed}")
    print(f"cuDNN Deterministic: {deterministic}, Benchmark: {benchmark}")

class nnUNetTrainer:
    @staticmethod
    def _collate_fn(batch):
        """Custom collate function to handle batch creation and resizing."""
        images, masks = zip(*batch)
        
        # Get target size from first image (assuming this is our desired patch size)
        target_size = (48, 128, 128)  # D, H, W
        
        # Resize if necessary and ensure contiguous tensors
        processed_images = []
        processed_masks = []
        
        for img, msk in zip(images, masks):
            if img.shape[-3:] != target_size:
                img = F.interpolate(
                    img.unsqueeze(0),
                    size=target_size,
                    mode='trilinear',
                    align_corners=False
                ).squeeze(0)
                
                msk = F.interpolate(
                    msk.unsqueeze(0).float(),
                    size=target_size,
                    mode='nearest'
                ).squeeze(0)
                
            processed_images.append(img.clone().contiguous())
            processed_masks.append(msk.clone().contiguous())
            
        # Stack the processed tensors
        stacked_images = torch.stack(processed_images, dim=0)
        stacked_masks = torch.stack(processed_masks, dim=0)
        
        return stacked_images, stacked_masks
    def __init__(self, config: nnUNetConfig):
        self.config = config
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
        
        # Initialize training components
        self.criterion = DC_and_CE_loss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.initial_lr,
            weight_decay=config.weight_decay
        )
        self.scaler = GradScaler()
        self.writer = SummaryWriter(f"runs/{config.experiment_name}_fold_{config.fold}")
        
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
        dataset = KiTS23Dataset(dataset_path, self.config)
        
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
            pin_memory=self.config.pin_memory,
            collate_fn=self._collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=self._collate_fn
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
                                # Calculate main loss
                                loss = self.criterion(outputs[0], targets)
                                
                                # Add deep supervision losses
                                ds_weight = 0.4
                                for aux_output in outputs[1:]:
                                    loss += ds_weight * self.criterion(aux_output, targets)
                                    
                                # Use main output for metrics
                                main_output = outputs[0]
                            else:
                                loss = self.criterion(outputs, targets)
                                main_output = outputs
                            
                            # Add auxiliary losses from deep supervision if in training mode
                            if isinstance(outputs, list) and len(outputs) > 1:
                                aux_weight = 0.4  # Weight for auxiliary losses
                                for aux_output in outputs[1:]:
                                    loss += aux_weight * self.criterion(aux_output, targets)
                        
                        # Backward pass
                        self.optimizer.zero_grad()
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        
                        # Update metrics
                        train_loss += loss.item()
                        # Calculate metrics using only the main output
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
            for batch_idx, (images, targets) in enumerate(pbar):
                try:
                    images, targets = images.to(self.device), targets.to(self.device)
                    
                    with autocast(enabled=self.config.use_amp):
                        outputs = self.model(images)
                        # Use only main output for validation
                        main_output = outputs[0] if isinstance(outputs, list) else outputs
                        
                        # Ensure output is at target resolution
                        if main_output.shape[-3:] != targets.shape[-3:]:
                            main_output = F.interpolate(
                                main_output,
                                size=targets.shape[-3:],
                                mode='trilinear',
                                align_corners=False
                            )
                        
                        loss = self.criterion(main_output, targets)
                    
                    current_dice = self._calculate_dice(main_output, targets)
                    val_loss += loss.item()
                    val_dice += current_dice
                    
                    # Update progress bar with current batch metrics
                    pbar.set_postfix({
                        'loss': loss.item(),
                        'dice': current_dice
                    })
                    
                except Exception as e:
                    print(f"\nError in validation batch {batch_idx}: {str(e)}")
                    raise e
        
        avg_loss = val_loss / len(val_loader)
        avg_dice = val_dice / len(val_loader)
        print(f"\nValidation complete - Average loss: {avg_loss:.4f}, Average dice: {avg_dice:.4f}")
        return avg_loss, avg_dice

    def _calculate_dice(self, outputs, targets):
        # Ensure outputs are at the same resolution as targets
        if outputs.shape[-3:] != targets.shape[-3:]:
            outputs = F.interpolate(
                outputs,
                size=targets.shape[-3:],
                mode='trilinear',
                align_corners=False
            )
            
        # Convert to binary predictions
        preds = torch.argmax(outputs, dim=1)
        if len(targets.shape) == len(preds.shape) + 1:
            targets = targets.squeeze(1)
            
        # Calculate dice coefficient
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        
        return (2. * intersection + 1e-5) / (union + 1e-5)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train')
    args = parser.parse_args()
    
    config = nnUNetConfig()
    config.resume_training = args.resume
    
    # Override config values with command line arguments if provided
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.num_epochs = args.epochs
    
    # Set seed for reproducibility before initializing anything else
    set_seed(config.seed, config.deterministic, config.benchmark_cudnn)
    
    trainer = nnUNetTrainer(config)
    trainer.train(config.data_dir)