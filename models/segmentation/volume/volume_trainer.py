import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import signal
import sys
import torchvision.transforms.functional as TF
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Fixed import
from models.segmentation.model import SegmentationModel
from losses_patch import WeightedDiceBCELoss
from models.segmentation.volume.dataset_volume import KiTS23VolumeDataset

logger = logging.getLogger(__name__)

class VolumeSegmentationTrainer:
    """Trainer for full-volume segmentation"""
    
    def __init__(self, config):
        """Initialize trainer with configuration"""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Configure CUDA memory management
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.95)
            torch.backends.cudnn.benchmark = True

        # Create model with correct number of input channels
        self.model = SegmentationModel(
            in_channels=config.in_channels,  # Determined by phase in config
            out_channels=1,  # Tumor probability
            features=config.features
        ).to(self.device)
        
        # Training settings
        self.current_epoch = 0
        self.start_epoch = 0
        self.best_val_dice = float('-inf')
        self.seed = 42

        # Initialize training components
        self.criterion = WeightedDiceBCELoss(
            pos_weight=torch.tensor([10.0]).to(self.device),
            dice_weight=1.0,
            bce_weight=1.0
        ).to(self.device)
        
        # Initialize optimizer with weight decay
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-5,  # Add L2 regularization
            eps=1e-8
        )
        
        # More aggressive learning rate scheduling
        self.scheduler = torch.optim.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.2,  # Larger LR reduction
            patience=3,   # Fewer epochs before reducing
            verbose=True,
            min_lr=1e-6
        )
        
        self.scaler = GradScaler()
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        # Load checkpoint if specified
        if config.checkpoint_path:
            self._load_checkpoint(config.checkpoint_path)
            
        # Print configuration
        phase_name = "Phase 3" if config.use_kidney_mask else "Phase 4"
        print(f"\nTraining Configuration ({phase_name})")
        print("-----------------------")
        print(f"PyTorch version: {torch.__version__}")
        print(f"Using device: {self.device}")
        print(f"Random seed: {self.seed}")
        
        print("\nSegmentation Model Configuration:")
        print(f"Input channels: {config.in_channels} ({'CT + kidney mask' if config.use_kidney_mask else 'CT only'})")
        print(f"Output channels: 1 (tumor probability)")
        print(f"Base features: {config.features}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    @staticmethod
    def _apply_augmentations(volume: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to volume"""
        # Random flip
        if random.random() > 0.5:
            volume = torch.flip(volume, [-1])  # Flip width
        if random.random() > 0.5:
            volume = torch.flip(volume, [-2])  # Flip height
            
        # Random rotation (-10 to 10 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            volume = TF.rotate(volume, angle)
            
        # Random Gaussian noise (small amplitude)
        if random.random() > 0.5:
            noise = torch.randn_like(volume) * 0.02
            volume = volume + noise
            
        # Random intensity scaling
        if random.random() > 0.5:
            scale = random.uniform(0.9, 1.1)
            volume = volume * scale
            
        return volume

    def train(self):
        """Run full training loop"""
        # Set random seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        
        # Create datasets
        dataset = KiTS23VolumeDataset(
            data_dir=self.config.data_dir,
            use_kidney_mask=self.config.use_kidney_mask,
            training=True,
            debug=self.config.debug
        )
        
        train_dataset, val_dataset = dataset.get_train_val_splits(
            val_ratio=0.2,
            seed=self.seed
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        try:
            signal.signal(signal.SIGINT, self._handle_interrupt)
            signal.signal(signal.SIGTERM, self._handle_interrupt)
            
            for epoch in range(self.start_epoch, self.config.num_epochs):
                self.current_epoch = epoch
                
                # Training
                train_loss, train_dice = self._train_epoch(train_loader)
                
                # Validation
                val_loss, val_dice = self._validate(val_loader)
                
                # Learning rate scheduling with early stopping
                self.scheduler.step(val_dice)
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Early stopping check
                if current_lr <= 1e-6:
                    print("\nLearning rate too low. Stopping training.")
                    break
                
                # Log metrics
                self._log_metrics(train_loss, train_dice, val_loss, val_dice, current_lr, epoch)
                
                # Save best model
                is_best = val_dice > self.best_val_dice
                if is_best:
                    self.best_val_dice = val_dice
                    
                self._save_checkpoint(
                    epoch,
                    {
                        'loss': val_loss,
                        'dice': val_dice,
                        'phase': 3 if self.config.use_kidney_mask else 4
                    },
                    is_best
                )
                
        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving checkpoint...")
            self._save_checkpoint(
                self.current_epoch,
                {'dice': self.best_val_dice},
                is_best=False
            )
            sys.exit(0)
            
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            raise
            
        self.writer.close()
        print(f"\nTraining completed! Best validation Dice: {self.best_val_dice:.4f}")

    def _train_epoch(self, train_loader):
        """Run one epoch of training"""
        self.model.train()
        train_loss = 0
        train_dice = 0
        n_batches = 0
        
        with tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}/{self.config.num_epochs}") as pbar:
            for batch in pbar:
                # Move data to device
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # Apply augmentations
                if self.training:
                    inputs = self._apply_augmentations(inputs)
                
                # Forward pass with autocast
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Backward pass with gradient clipping
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Calculate metrics
                with torch.no_grad():
                    probs = torch.sigmoid(outputs)
                    pred_mask = (probs > 0.5).float()
                    dice_score = self._calculate_dice(pred_mask, targets)
                
                # Update statistics
                train_loss += loss.item()
                train_dice += dice_score
                n_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'dice': f"{dice_score:.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
                
                # Clear cache periodically
                if n_batches % 10 == 0:
                    torch.cuda.empty_cache()
        
        return (train_loss / n_batches if n_batches > 0 else float('inf'),
                train_dice / n_batches if n_batches > 0 else 0)

    def _validate(self, val_loader):
        """Run validation"""
        self.model.eval()
        val_loss = 0
        val_dice = 0
        n_volumes = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                try:
                    # Get full volume
                    inputs = batch['input'].to(self.device)
                    targets = batch['target'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    
                    # Calculate metrics
                    probs = torch.sigmoid(outputs)
                    pred_mask = (probs > 0.5).float()
                    dice_score = self._calculate_dice(pred_mask, targets)
                    
                    val_loss += loss.item()
                    val_dice += dice_score
                    n_volumes += 1
                    
                    if self.config.debug and n_volumes == 1:
                        self._visualize_full_volume(inputs, targets, pred_mask, batch['case_id'][0])
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.error(f"\nCUDA OOM during validation. Skipping volume.")
                        torch.cuda.empty_cache()
                        continue
                    raise e
        
        return (val_loss / n_volumes if n_volumes > 0 else float('inf'),
                val_dice / n_volumes if n_volumes > 0 else 0)

    def _visualize_full_volume(self, inputs, targets, predictions, case_id):
        """Create visualization of full-volume predictions"""
        # Get middle slices
        slice_idx = inputs.shape[2] // 2
        
        n_plots = 4 if self.config.use_kidney_mask else 3
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
        
        # Plot CT
        axes[0].imshow(inputs[0, 0, slice_idx].cpu(), cmap='gray')
        axes[0].set_title('CT')
        axes[0].axis('off')
        
        if self.config.use_kidney_mask:
            # Plot kidney mask
            axes[1].imshow(inputs[0, 1, slice_idx].cpu(), cmap='gray')
            axes[1].set_title('Kidney Mask')
            axes[1].axis('off')
            
            # Plot ground truth
            axes[2].imshow(targets[0, 0, slice_idx].cpu(), cmap='gray')
            axes[2].set_title('Ground Truth')
            axes[2].axis('off')
            
            # Plot prediction
            axes[3].imshow(predictions[0, 0, slice_idx].cpu(), cmap='gray')
            axes[3].set_title(f'Prediction (Case {case_id})')
            axes[3].axis('off')
        else:
            # Plot ground truth
            axes[1].imshow(targets[0, 0, slice_idx].cpu(), cmap='gray')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            # Plot prediction
            axes[2].imshow(predictions[0, 0, slice_idx].cpu(), cmap='gray')
            axes[2].set_title(f'Prediction (Case {case_id})')
            axes[2].axis('off')
        
        # Add to tensorboard
        self.writer.add_figure('Full Volume Predictions', fig, self.current_epoch)
        plt.close(fig)

    def _calculate_dice(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate Dice score between prediction and target"""
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        return (2 * intersection + 1e-5) / (union + 1e-5)

    def _log_metrics(self, train_loss, train_dice, val_loss, val_dice, lr, epoch):
        """Log epoch metrics to console and tensorboard"""
        # Add scalars to tensorboard
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        self.writer.add_scalar('Loss/Val', val_loss, epoch)
        self.writer.add_scalar('Dice/Train', train_dice, epoch)
        self.writer.add_scalar('Dice/Val', val_dice, epoch)
        self.writer.add_scalar('LearningRate', lr, epoch)
        
        # Print to console
        print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")
        print(f"Learning rate: {lr:.2e}")
        if val_dice > self.best_val_dice:
            print("New best validation Dice!")

    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save training checkpoint"""
        phase = "phase3" if self.config.use_kidney_mask else "phase4"
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_dice': self.best_val_dice,
            'metrics': metrics,
            'config': {
                'use_kidney_mask': self.config.use_kidney_mask,
                'in_channels': self.config.in_channels,
                'training_phase': 3 if self.config.use_kidney_mask else 4,
                'seed': self.seed
            }
        }
        
        # Save latest checkpoint
        latest_path = self.config.checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)
        
        # Save best model
        if is_best:
            best_path = self.config.checkpoint_dir / f"best_model_{phase}_dice_{metrics['dice']:.4f}.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model with Dice score: {metrics['dice']:.4f}")

    def _load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        if not Path(checkpoint_path).exists():
            logger.error(f"No checkpoint found at {checkpoint_path}")
            return
            
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        # Get model state dict
        state_dict = checkpoint['model_state_dict']
        
        # Handle input channel mismatch for Phase 3
        if self.config.use_kidney_mask:
            print("Adapting Phase 1 weights for Phase 3 (2-channel input)...")
            # Find all encoder convolution layers
            for key in list(state_dict.keys()):
                # Only process conv weight tensors (not bias)
                if 'weight' in key and len(state_dict[key].shape) == 5:  # 5D: [out_ch, in_ch, D, H, W]
                    conv_weight = state_dict[key]
                    if conv_weight.shape[1] == 1:  # If single channel input
                        # Duplicate channel and scale second channel
                        new_conv_weight = conv_weight.repeat(1, 2, 1, 1, 1)
                        new_conv_weight[:, 1] *= 0.5
                        state_dict[key] = new_conv_weight
                        print(f"Adapted {key}: {conv_weight.shape} -> {new_conv_weight.shape}")
        
        # Load modified state dict
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            raise
        
        # Get checkpoint phase
        checkpoint_phase = checkpoint['config'].get('training_phase', 1)
        
        # Special handling for Phase 1 -> Phase 3 transition
        if checkpoint_phase == 1 and self.config.use_kidney_mask:
            print("Loading Phase 1 → Phase 3; reinitializing optimizer/scheduler/scaler")
            # Start fresh with expanded model weights
            self.start_epoch = 0
            self.best_val_dice = float('-inf')
            # Initialize fresh optimizer and scheduler 
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=1e-5,
                eps=1e-8
            )
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.2,
                patience=3,
                verbose=True,
                min_lr=1e-6
            )
            self.scaler = GradScaler()
            print("Initialized fresh optimizer and scheduler.")
        else:
            # Same phase or Phase 3->4: resume everything
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_dice = checkpoint['best_val_dice']
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        print(f"Starting training from epoch {self.start_epoch}")

    def _handle_interrupt(self, signum, frame):
        """Handle interrupt signals gracefully"""
        print("\nInterrupt received. Saving checkpoint before exiting...")
        self._save_checkpoint(
            self.current_epoch,
            {'dice': self.best_val_dice},
            is_best=False
        )
        print("Checkpoint saved. Exiting gracefully.")
        sys.exit(0)