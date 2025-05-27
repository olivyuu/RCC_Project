import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import numpy as np
from tqdm import tqdm
import signal
import sys
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler  # Add explicit import

from model import nnUNetv2
from dataset_volume import KiTS23VolumeDataset
from losses import DC_and_BCE_loss

class nnUNetVolumeTrainer:
    def __init__(self, config, debug_logger=None):
        self.config = config
        self.debug_logger = debug_logger

        # Initialize preprocessor early
        from preprocessing.preprocessor import KiTS23Preprocessor
        self.preprocessor = KiTS23Preprocessor(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create checkpoint directory
        self.config.checkpoint_dir.mkdir(exist_ok=True)
        
        # Training state initialization
        self.start_epoch = 0
        self.best_val_dice = float('-inf')
        self.patience_counter = 0
        self.current_epoch = 0
        self.frozen_encoder_blocks = 2  # Start with fewer blocks frozen for more aggressive training
        
        # Initialize model with 2 input channels
        self.model = nnUNetv2(
            in_channels=2,  # Changed to 2 for image + kidney mask
            out_channels=2,  # Binary segmentation (background, tumor)
            features=config.features
        ).to(self.device)
        
        # Load patch-trained weights if specified
        if config.transfer_learning and config.patch_weights_path:
            self._load_patch_weights()
        
        # Initialize training components with new combined loss
        self.criterion = DC_and_BCE_loss()
        
        # Use AdamW optimizer with higher initial learning rate
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.initial_lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler with longer cycles
        self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(  # Use lr_scheduler namespace
            self.optimizer,
            T_0=20,  # Longer initial cycle for better exploration
            T_mult=1,  # Keep consistent cycle length
            eta_min=1e-5  # Higher minimum learning rate
        )
        
        self.scaler = GradScaler()
        self.writer = SummaryWriter(f"runs/{config.experiment_name}_volume_fold_{config.fold}")
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
        
        if config.resume_training:
            self._load_checkpoint()
    
    def _load_patch_weights(self):
        """Load weights from patch-trained model."""
        print(f"Loading patch-trained weights from {self.config.patch_weights_path}")
        checkpoint = torch.load(self.config.patch_weights_path)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Modify first conv layer to accept 2 channels
        old_conv = state_dict['down_blocks.0.conv_block.0.conv.weight']
        new_conv = torch.zeros(old_conv.shape[0], 2, *old_conv.shape[2:])
        new_conv[:, 0] = old_conv.squeeze(1)  # Copy weights for image channel
        new_conv[:, 1] = old_conv.squeeze(1).mean(dim=0)  # Initialize kidney mask channel
        state_dict['down_blocks.0.conv_block.0.conv.weight'] = new_conv

        self.model.load_state_dict(state_dict, strict=False)
        self._freeze_encoder_blocks(self.frozen_encoder_blocks)
    
    def _freeze_encoder_blocks(self, num_blocks):
        """Freeze specified number of encoder blocks."""
        print(f"Freezing {num_blocks} encoder blocks")
        for name, param in self.model.named_parameters():
            # Freeze specified number of encoder blocks
            for i in range(num_blocks):
                if f'down_blocks.{i}' in name:
                    param.requires_grad = False
        self.frozen_encoder_blocks = num_blocks
    
    def _unfreeze_one_block(self):
        """Unfreeze the next encoder block."""
        if self.frozen_encoder_blocks > 0:
            self.frozen_encoder_blocks -= 1
            print(f"Unfreezing encoder block {self.frozen_encoder_blocks}")
            # Unfreeze the specific block
            for name, param in self.model.named_parameters():
                if f'down_blocks.{self.frozen_encoder_blocks}' in name:
                    param.requires_grad = True

    def _get_supervision_weights(self, epoch):
        """Get progressive weights for deep supervision outputs."""
        # Start with emphasis on detection (deeper layers)
        # Gradually shift to segmentation (shallower layers)
        progress = min(epoch / (self.config.num_epochs * 0.5), 1.0)  # Transition over 50% of training
        
        # More aggressive weight transition
        weights = torch.tensor([
            0.3 - 0.2 * progress,  # Final output (detection) - less emphasis
            0.2 - 0.1 * progress,  # Deep3
            0.2 + 0.1 * progress,  # Deep2
            0.3 + 0.2 * progress   # Deep1 (segmentation) - more emphasis
        ])
        
        return weights.to(self.device)

    def train(self, dataset_path: str):
        # Create dataset and loaders
        dataset = KiTS23VolumeDataset(
            dataset_path,
            self.config,
            preprocessor=self.preprocessor,
            train=True,
            preprocess=self.config.preprocess
        )
        
        # Split and create dataloaders
        val_size = int(len(dataset) * self.config.validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.seed)
        )
        
        train_sampler = self._create_weighted_sampler(train_dataset)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.vol_batch_size,
            sampler=train_sampler,
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
                
                # More frequent unfreezing (every 1/8th of training)
                if epoch > 0 and epoch % (self.config.num_epochs // 8) == 0:
                    self._unfreeze_one_block()
                
                # Get supervision weights for this epoch
                supervision_weights = self._get_supervision_weights(epoch)
                
                # Training phase
                self.model.train()
                train_loss = 0
                train_dice = 0
                self.optimizer.zero_grad()
                
                with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}") as pbar:
                    for batch_idx, (images, targets) in enumerate(pbar):
                        images = images.to(self.device)
                        targets = targets.to(self.device)
                        
                        with autocast(enabled=self.config.use_amp):
                            outputs = self.model(images)
                            if isinstance(outputs, list):
                                # Progressive learning with weighted deep supervision
                                loss = 0
                                for idx, (output, weight) in enumerate(zip(outputs, supervision_weights)):
                                    if idx == 0:
                                        loss += weight * self.criterion(output, targets)
                                    else:
                                        scaled_target = F.interpolate(
                                            targets.float(),
                                            size=output.shape[2:],
                                            mode='nearest'
                                        )
                                        loss += weight * self.criterion(output, scaled_target)
                                main_output = outputs[0]
                            else:
                                loss = self.criterion(outputs, targets)
                                main_output = outputs
                            
                            loss = loss / self.config.vol_gradient_accumulation_steps
                        
                        self.scaler.scale(loss).backward()
                        
                        if (batch_idx + 1) % self.config.vol_gradient_accumulation_steps == 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.optimizer.zero_grad()
                        
                        train_loss += loss.item() * self.config.vol_gradient_accumulation_steps
                        dice = self._calculate_dice(main_output, targets)
                        train_dice += dice
                        
                        pbar.set_postfix({
                            'loss': loss.item() * self.config.vol_gradient_accumulation_steps,
                            'dice': dice,
                            'lr': self.optimizer.param_groups[0]['lr']
                        })
                
                self.scheduler.step()
                
                train_loss /= len(train_loader)
                train_dice /= len(train_loader)
                
                val_loss, val_dice = self._validate(val_loader)
                
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
    
    def _create_weighted_sampler(self, dataset):
        """Create a weighted sampler to focus on challenging cases."""
        weights = []
        for idx in range(len(dataset)):
            _, target = dataset[idx]
            tumor_size = (target > 0).sum().item()
            total_size = target.numel()
            ratio = tumor_size / total_size
            weight = 1.0 / (ratio + 1e-5)
            weights.append(weight)
        
        weights = torch.FloatTensor(weights)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights,
            len(weights),
            replacement=True
        )
        return sampler
    
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
        if outputs.shape != targets.shape:
            targets = F.interpolate(
                targets.float(),
                size=outputs.shape[-3:],
                mode='nearest'
            )
        
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
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_dice': self.best_val_dice,
            'patience_counter': self.patience_counter,
            'metrics': metrics,
            'frozen_encoder_blocks': self.frozen_encoder_blocks
        }
        
        if self.config.save_latest:
            latest_path = self.config.checkpoint_dir / "volume_latest.pth"
            torch.save(checkpoint, latest_path)
        
        if is_best:
            best_path = self.config.checkpoint_dir / f"volume_best_dice_{metrics['dice']:.4f}.pth"
            torch.save(checkpoint, best_path)
        
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
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_dice = checkpoint['best_val_dice']
        self.patience_counter = checkpoint['patience_counter']
        self.frozen_encoder_blocks = checkpoint.get('frozen_encoder_blocks', 2)
        
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