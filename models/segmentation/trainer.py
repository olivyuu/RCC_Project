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

def check_tensor(tensor, name="", check_grad=False):
    """Debug helper to check tensor properties"""
    if tensor is None:
        print(f"{name} is None!")
        return False
        
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()
    
    if has_nan or has_inf:
        print(f"\nWarning: {name} contains NaN: {has_nan} or Inf: {has_inf}")
        print(f"Shape: {tensor.shape}")
        print(f"Range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
        print(f"Mean: {tensor.mean().item():.4f}, Std: {tensor.std().item():.4f}")
        
        if check_grad and tensor.grad is not None:
            grad = tensor.grad
            print(f"Gradient range: [{grad.min().item():.4f}, {grad.max().item():.4f}]")
            print(f"Gradient mean: {grad.mean().item():.4f}, std: {grad.std().item():.4f}")
        
        return False
    return True

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
            
            # Check each parameter
            check_tensor(param, f"Parameter {name}", check_grad=True)
        
        self.stats['gradient_norms'].append(grad_norm)
        self.stats['weight_norms'].append(weight_norm)
        
        # Check outputs and targets
        check_tensor(outputs, "Model outputs")
        check_tensor(targets, "Targets")
        
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
                    f.write(f"Weight mean: {param.mean().item():.4f}, std: {param.std().item():.4f}\n")
                    f.write(f"Gradient range: [{param.grad.min().item():.4f}, {param.grad.max().item():.4f}]\n")
                    f.write(f"Gradient mean: {param.grad.mean().item():.4f}, std: {param.grad.std().item():.4f}\n")
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
        
        # Initialize debug stats
        self.debug_stats = DebugStats(config.log_dir)
        
        # Create checkpoint directory
        self.config.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize model with He initialization
        self.model = nnUNetv2(
            in_channels=2,
            out_channels=1,
            features=config.features
        ).to(self.device)
        
        # Initialize weights properly
        self._initialize_weights()
        
        # Initialize training components with stable settings
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
                self.model.train()
                train_loss = 0
                train_dice = 0
                valid_batches = 0
                
                with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}") as pbar:
                    for batch_idx, (images, targets) in enumerate(pbar):
                        images = images.to(self.device)
                        targets = targets.to(self.device)
                        
                        # Check inputs
                        check_tensor(images, "Input images")
                        check_tensor(targets, "Target masks")
                        
                        # Forward pass without mixed precision initially
                        outputs = self.model(images)
                        if isinstance(outputs, (list, tuple)):
                            outputs = outputs[-1]
                        
                        # Debug pre-interpolation
                        if batch_idx % 10 == 0:
                            print(f"\nPre-interpolation shape: {outputs.shape}")
                            print(f"Pre-interpolation range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
                            check_tensor(outputs, "Pre-interpolation outputs")
                        
                        if outputs.shape[-3:] != targets.shape[-3:]:
                            outputs = F.interpolate(
                                outputs.float(),
                                size=targets.shape[-3:],
                                mode='trilinear',
                                align_corners=False
                            )
                            
                            if batch_idx % 10 == 0:
                                print(f"Post-interpolation shape: {outputs.shape}")
                                print(f"Post-interpolation range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
                                check_tensor(outputs, "Post-interpolation outputs")
                        
                        # Calculate loss without sigmoid activation
                        loss = self.criterion(outputs, targets)
                        
                        if torch.isnan(loss) or torch.isinf(loss):
                            print("\nWarning: Invalid loss value detected!")
                            print(f"Loss: {loss.item()}")
                            continue
                        
                        # Backward pass with gradient monitoring
                        self.optimizer.zero_grad()
                        loss.backward()
                        
                        # Check gradients before clipping
                        if batch_idx % 10 == 0:
                            print("\nGradient norms before clipping:")
                            for name, param in self.model.named_parameters():
                                if param.grad is not None:
                                    grad_norm = param.grad.norm().item()
                                    print(f"{name}: {grad_norm:.4f}")
                                    check_tensor(param.grad, f"Gradient for {name}")
                        
                        # Clip gradients
                        total_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.max_grad_norm
                        )
                        
                        if batch_idx % 10 == 0:
                            print(f"Total gradient norm after clipping: {total_norm:.4f}")
                        
                        self.optimizer.step()
                        
                        # Calculate metrics
                        with torch.no_grad():
                            outputs_sigmoid = torch.sigmoid(outputs)  # Apply sigmoid after training
                            dice = self._calculate_dice(outputs_sigmoid.detach(), targets)
                            
                            if not torch.isnan(dice) and not torch.isinf(dice):
                                train_loss += loss.item()
                                train_dice += dice
                                valid_batches += 1
                                
                                self.debug_stats.log_batch(
                                    epoch, batch_idx, self.model,
                                    loss, dice, outputs_sigmoid, targets
                                )
                        
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
                
                # Calculate loss and metrics
                loss = self.criterion(outputs, targets)
                outputs_sigmoid = torch.sigmoid(outputs)
                dice = self._calculate_dice(outputs_sigmoid, targets)
                
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
        # Threshold predictions
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
            
            # Additional prediction statistics
            print(f"Prediction stats - Mean: {outputs.mean().item():.4f}, Std: {outputs.std().item():.4f}")
            print(f"Target stats - Mean: {targets.mean().item():.4f}, Std: {targets.std().item():.4f}")
        
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