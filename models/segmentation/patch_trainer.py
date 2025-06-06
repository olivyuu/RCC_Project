import multiprocessing as mp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler
import torch.cuda
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import numpy as np
from tqdm import tqdm
import signal
import sys
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import logging
import random

from models.segmentation.model import SegmentationModel
from dataset_patch import KiTS23PatchDataset, worker_init_fn
from losses_patch import WeightedDiceBCELoss
from dataset_volume import KiTS23VolumeDataset

logger = logging.getLogger(__name__)

class PatchSegmentationTrainer:
    """Trainer for patch-based segmentation model"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Configure CUDA memory management
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.95)
            torch.backends.cudnn.benchmark = True
        
        # Create model
        self.model = SegmentationModel(
            in_channels=1,  # CT only
            out_channels=1,  # Single channel output (tumor)
            features=self.config.features  # Base feature channels
        ).to(self.device)
        
        # Training settings
        self.max_grad_norm = 5.0
        self.batch_size = 2
        self.tumor_only_prob = 0.7
        self.patch_size = (64, 128, 128)
        self.num_workers = 4
        self.start_epoch = 0
        self.current_epoch = 0
        self.best_val_dice = float('-inf')
        self.seed = 42  # Global seed for reproducibility
        
        # Create checkpoint directory
        self.config.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize training components
        self.criterion = WeightedDiceBCELoss(
            pos_weight=torch.tensor([10.0]).to(self.device),  # Move to device
            dice_weight=1.0,
            bce_weight=1.0
        )
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=5e-5,
            weight_decay=1e-5,
            eps=1e-8
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-6
        )
        
        self.scaler = GradScaler()
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        # Load checkpoint if specified
        if config.checkpoint_path:
            self._load_checkpoint(config.checkpoint_path)
        elif config.resume_training:
            self._load_latest_checkpoint()

    def train(self, dataset_path):
        """Main training loop"""
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass

        # Set random seeds for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Create volume dataset and split train/val
        volume_dataset = KiTS23VolumeDataset(dataset_path, self.config)
        train_dataset, val_dataset = volume_dataset.get_train_val_splits(val_ratio=0.2, seed=self.seed)
        
        # Create patch datasets with split volumes
        train_patch_dataset = KiTS23PatchDataset(
            dataset_path,  # Use path directly
            patch_size=self.patch_size,
            tumor_only_prob=self.tumor_only_prob,
            use_kidney_mask=self.config.use_kidney_mask,
            min_kidney_voxels=self.config.min_kidney_voxels if hasattr(self.config, 'min_kidney_voxels') else 100,
            kidney_patch_overlap=self.config.kidney_patch_overlap if hasattr(self.config, 'kidney_patch_overlap') else 0.5,
            debug=self.config.debug
        )
        
        val_patch_dataset = KiTS23PatchDataset(
            dataset_path,  # Use path directly
            patch_size=self.patch_size,
            tumor_only_prob=0.5,  # Equal sampling for validation
            use_kidney_mask=self.config.use_kidney_mask,
            min_kidney_voxels=self.config.min_kidney_voxels if hasattr(self.config, 'min_kidney_voxels') else 100,
            kidney_patch_overlap=self.config.kidney_patch_overlap if hasattr(self.config, 'kidney_patch_overlap') else 0.5,
            debug=False
        )
        
        # Assign split volume paths
        train_patch_dataset.volume_paths = train_dataset.volume_paths
        val_patch_dataset.volume_paths = val_dataset.volume_paths
        
        # Create data loaders with worker initialization
        train_loader = DataLoader(
            train_patch_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            worker_init_fn=worker_init_fn,  # Initialize worker RNG
        )
        
        val_loader = DataLoader(
            val_patch_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            worker_init_fn=worker_init_fn,  # Initialize worker RNG
        )

        print(f"\nTraining Configuration ({self.config.get_phase_name()})")
        print("-----------------------")
        print(f"PyTorch version: {torch.__version__}")
        print(f"Using device: {self.device}")
        print(f"Random seed: {self.seed}")
        
        print("\nSegmentation Model Configuration:")
        print(f"Input channels: 1 (CT only)")
        print(f"Output channels: 1 (tumor probability)")
        print(f"Base features: {self.config.features}")
        print(f"Encoder depths: 32 → 64 → 128 → 256")
        print(f"Bottleneck features: 512")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        if self.config.checkpoint_path:
            print(f"Loading checkpoint: {self.config.checkpoint_path}")
        
        if self.config.use_kidney_mask:
            print("Phase 2 parameters:")
            print(f"  min_kidney_voxels: {self.config.min_kidney_voxels}")
            print(f"  kidney_patch_overlap: {self.config.kidney_patch_overlap}")
            
        print("\nModel Configuration:")
        print(f"Patch size: {self.patch_size}")
        print(f"Batch size: {self.batch_size}")
        print(f"Number of workers: {self.num_workers}")
        print(f"Gradient clipping: {self.max_grad_norm}")
        print(f"Tumor sampling probability: {self.tumor_only_prob}")
        
        print("\nLoss Configuration:")
        print(f"Positive class weight: {self.criterion.bce.pos_weight.item()}")
        print(f"Dice loss weight: {self.criterion.dice_weight}")
        print(f"BCE loss weight: {self.criterion.bce_weight}")
        
        print("\nOptimizer Configuration:")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"Weight decay: {self.optimizer.param_groups[0]['weight_decay']}")
        
        if self.config.use_kidney_mask:
            print("\nPhase 2 Configuration:")
            print(f"Minimum kidney voxels: {self.config.min_kidney_voxels}")
            print(f"Required kidney overlap: {self.config.kidney_patch_overlap*100:.1f}%")
            
        if self.config.debug:
            print("\nDebug mode enabled - will show additional output and visualizations")

        try:
            signal.signal(signal.SIGINT, self._handle_interrupt)
            signal.signal(signal.SIGTERM, self._handle_interrupt)
            
            for epoch in range(self.start_epoch, self.config.num_epochs):
                self.current_epoch = epoch
                
                # Training phase
                train_loss, train_stats = self._train_epoch(train_loader)
                
                # Validation phase
                val_loss, val_dice = self._validate(val_loader)
                
                # Learning rate scheduling
                self.scheduler.step(val_dice)
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Log metrics
                self._log_epoch(train_loss, train_stats, val_loss, val_dice, current_lr, epoch)
                
                # Save best model
                is_best = val_dice > self.best_val_dice
                if is_best:
                    self.best_val_dice = val_dice
                
                self._save_checkpoint(
                    epoch,
                    {
                        'loss': val_loss,
                        'dice': val_dice,
                        'phase': 2 if self.config.use_kidney_mask else 1
                    },
                    is_best=is_best
                )
                
                # Visualize predictions periodically
                if self.config.debug and epoch % 5 == 0:
                    self._visualize_predictions(val_loader, epoch)

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
            raise
        
        self.writer.close()
        print("\nTraining completed!")
        print(f"Best validation Dice score: {self.best_val_dice:.4f}")

    def _train_epoch(self, train_loader):
        """Run one epoch of training"""
        self.model.train()
        train_loss = 0
        valid_batches = 0
        total_norm = 0
        
        # Track statistics
        tumor_ratios = []
        kidney_ratios = []  # Phase 2: Track kidney coverage
        mean_probs = []
        max_probs = []
        dice_scores = []
        
        with tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}/{self.config.num_epochs}") as pbar:
            for batch_idx, (images, tumor_masks, kidney_masks) in enumerate(pbar):
                # Move data to device
                images = images.to(self.device)
                tumor_masks = tumor_masks.to(self.device)
                kidney_masks = kidney_masks.to(self.device)
                
                # Use CT image directly (no kidney channel)
                inputs = images[:, 0:1].float()
                
                # Forward pass
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    
                    # Ensure output shape matches target
                    if outputs.shape[2:] != tumor_masks.shape[2:]:
                        outputs = F.interpolate(
                            outputs,
                            size=tumor_masks.shape[2:],
                            mode='trilinear',
                            align_corners=False
                        )
                    
                    # Compute loss
                    loss = self.criterion(outputs, tumor_masks, kidney_masks)
                
                # Backward pass with gradient clipping
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Compute metrics
                with torch.no_grad():
                    stats = self.criterion.log_stats(outputs, tumor_masks, kidney_masks)
                    tumor_ratios.append(stats['tumor_ratio'])
                    if 'kidney_ratio' in stats:
                        kidney_ratios.append(stats['kidney_ratio'])
                    mean_probs.append(stats['mean_prob'])
                    max_probs.append(stats['max_prob'])
                    dice_scores.append(stats['dice_score'])
                    
                    if batch_idx % 50 == 0:
                        self._log_batch_stats(stats, batch_idx, self.current_epoch)
                    
                    train_loss += loss.item()
                    valid_batches += 1
                
                # Update progress bar
                postfix = {
                    'loss': f"{loss.item():.4f}",
                    'tumor_ratio': f"{stats['tumor_ratio']:.4%}",
                    'dice': f"{stats['dice_score']:.4f}",
                    'grad_norm': f"{total_norm:.4f}"
                }
                if kidney_ratios:
                    postfix['kidney_ratio'] = f"{stats['kidney_ratio']:.4%}"
                pbar.set_postfix(postfix)
                
                # Clear cache periodically
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
        
        # Calculate epoch statistics
        avg_stats = {
            'loss': train_loss / valid_batches if valid_batches > 0 else float('inf'),
            'tumor_ratio': np.mean(tumor_ratios),
            'mean_prob': np.mean(mean_probs),
            'max_prob': np.mean(max_probs),
            'dice_score': np.mean(dice_scores)
        }
        if kidney_ratios:
            avg_stats['kidney_ratio'] = np.mean(kidney_ratios)
        
        return avg_stats['loss'], avg_stats

    def _validate(self, val_loader):
        """Validate on patches to avoid OOM"""
        self.model.eval()
        val_loss = 0
        val_dice = 0
        valid_batches = 0
        
        with tqdm(val_loader, desc="Validating") as pbar:
            for images, tumor_masks, kidney_masks in pbar:
                try:
                    # Move data to device
                    images = images.to(self.device)
                    tumor_masks = tumor_masks.to(self.device)
                    kidney_masks = kidney_masks.to(self.device)
                    
                    # Use CT image only
                    inputs = images[:, 0:1].float()
                    
                    with torch.no_grad():
                        # Forward pass
                        outputs = self.model(inputs)
                        
                        # Ensure output shape matches target
                        if outputs.shape[2:] != tumor_masks.shape[2:]:
                            outputs = F.interpolate(
                                outputs,
                                size=tumor_masks.shape[2:],
                                mode='trilinear',
                                align_corners=False
                            )
                        
                        # Compute loss and metrics
                        loss = self.criterion(outputs, tumor_masks, kidney_masks)
                        stats = self.criterion.log_stats(outputs, tumor_masks, kidney_masks)
                        
                        val_loss += loss.item()
                        val_dice += stats['dice_score']
                        valid_batches += 1
                        
                        pbar.set_postfix({
                            'loss': f"{loss.item():.4f}",
                            'dice': f"{stats['dice_score']:.4f}"
                        })
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\nCUDA OOM during validation. Skipping batch.")
                        torch.cuda.empty_cache()
                        continue
                    raise e
        
        return (val_loss / valid_batches if valid_batches > 0 else float('inf'),
                val_dice / valid_batches if valid_batches > 0 else 0)

    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_dice': self.best_val_dice,
            'metrics': metrics,
            'config': {
                'patch_size': self.patch_size,
                'batch_size': self.batch_size,
                'tumor_only_prob': self.tumor_only_prob,
                'use_kidney_mask': self.config.use_kidney_mask,
                'min_kidney_voxels': getattr(self.config, 'min_kidney_voxels', None),
                'kidney_patch_overlap': getattr(self.config, 'kidney_patch_overlap', None),
                'training_phase': 2 if self.config.use_kidney_mask else 1,
                'seed': self.seed
            }
        }
        
        latest_path = self.config.checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)
        
        if is_best:
            phase_str = "phase2" if self.config.use_kidney_mask else "phase1"
            best_path = self.config.checkpoint_dir / f"best_model_{phase_str}_dice_{metrics['dice']:.4f}.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model with Dice: {metrics['dice']:.4f}")

    def _load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        if not Path(checkpoint_path).exists():
            logger.error(f"No checkpoint found at {checkpoint_path}")
            return
        
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Verify phase compatibility
        checkpoint_phase = checkpoint['config'].get('training_phase', 1)
        current_phase = 2 if self.config.use_kidney_mask else 1
        
        if checkpoint_phase != current_phase:
            logger.warning(
                f"Loading Phase {checkpoint_phase} checkpoint in Phase {current_phase} training.\n"
                "Starting fresh from epoch 0 for new phase."
            )
            # Reset counters but keep model weights for new phase
            self.start_epoch = 0
            self.best_val_dice = float('-inf')
            # Initialize fresh optimizer and scheduler
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=1e-5,  # Lower learning rate for phase transition
                weight_decay=1e-5,
                eps=1e-8
            )
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, 
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True,
                min_lr=1e-6
            )
            self.scaler = GradScaler()
        else:
            # Continue from saved epoch if same phase
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_dice = checkpoint['best_val_dice']
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load configuration if available
        if 'config' in checkpoint:
            self.patch_size = checkpoint['config']['patch_size']
            self.batch_size = checkpoint['config']['batch_size']
            self.tumor_only_prob = checkpoint['config']['tumor_only_prob']
            self.seed = checkpoint['config'].get('seed', 42)
            
            # Verify Phase 2 parameters match if applicable
            if current_phase == 2:
                if checkpoint.get('min_kidney_voxels') != self.config.min_kidney_voxels:
                    logger.warning(
                        f"Checkpoint min_kidney_voxels ({checkpoint.get('min_kidney_voxels')}) "
                        f"differs from current setting ({self.config.min_kidney_voxels})"
                    )
                if checkpoint.get('kidney_patch_overlap') != self.config.kidney_patch_overlap:
                    logger.warning(
                        f"Checkpoint kidney_patch_overlap ({checkpoint.get('kidney_patch_overlap')}) "
                        f"differs from current setting ({self.config.kidney_patch_overlap})"
                    )
        
        print(f"Starting training from epoch {self.start_epoch}")
        if current_phase == 2:
            print("Phase 2 parameters:")
            print(f"  min_kidney_voxels: {self.config.min_kidney_voxels}")
            print(f"  kidney_patch_overlap: {self.config.kidney_patch_overlap}")
        
        # Print learning rate
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")

    def _load_latest_checkpoint(self):
        """Load latest training checkpoint"""
        latest_path = self.config.checkpoint_dir / "latest.pth"
        if latest_path.exists():
            self._load_checkpoint(str(latest_path))

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

    def _visualize_predictions(self, val_loader, epoch):
        """Create visualization of current predictions"""
        self.model.eval()
        
        # Get first validation batch
        images, tumor_masks, kidney_masks = next(iter(val_loader))
        images = images.to(self.device)
        tumor_masks = tumor_masks.to(self.device)
        kidney_masks = kidney_masks.to(self.device)
        
        with torch.no_grad():
            # Get predictions
            inputs = images[:, 0:1].float()
            outputs = self.model(inputs)
            
            # Ensure output shape matches target
            if outputs.shape[2:] != tumor_masks.shape[2:]:
                outputs = F.interpolate(
                    outputs,
                    size=tumor_masks.shape[2:],
                    mode='trilinear',
                    align_corners=False
                )
                
            # Get probabilities
            probs = torch.sigmoid(outputs)
            
            # Get middle slice for visualization
            slice_idx = probs.shape[2] // 2
            
            # Create figure with subplots (4 in Phase 2, 3 in Phase 1)
            n_plots = 4 if self.config.use_kidney_mask else 3
            fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
            
            # Plot CT
            axes[0].imshow(images[0, 0, slice_idx].cpu(), cmap='gray')
            axes[0].set_title('CT')
            axes[0].axis('off')
            
            # Plot ground truth tumor
            axes[1].imshow(tumor_masks[0, 0, slice_idx].cpu(), cmap='gray')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            if self.config.use_kidney_mask:
                # Plot kidney mask
                axes[2].imshow(kidney_masks[0, 0, slice_idx].cpu(), cmap='gray')
                axes[2].set_title('Kidney Mask')
                axes[2].axis('off')
                
                # Plot prediction
                axes[3].imshow(probs[0, 0, slice_idx].cpu(), cmap='gray')
                axes[3].set_title(f'Prediction (Epoch {epoch+1})')
                axes[3].axis('off')
            else:
                # Plot prediction
                axes[2].imshow(probs[0, 0, slice_idx].cpu(), cmap='gray')
                axes[2].set_title(f'Prediction (Epoch {epoch+1})')
                axes[2].axis('off')
            
            # Add to tensorboard
            self.writer.add_figure('Predictions', fig, epoch)
            plt.close(fig)

    def _log_batch_stats(self, stats: dict, batch_idx: int, epoch: int):
        """Log detailed batch statistics"""
        step = epoch * 1000 + batch_idx
        
        # Log all statistics
        for key, value in stats.items():
            self.writer.add_scalar(f'Batch/{key}', value, step)

    def _log_epoch(self, train_loss, train_stats, val_loss, val_dice, lr, epoch):
        """Log epoch-level metrics"""
        # Add scalars to tensorboard
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        self.writer.add_scalar('Loss/Val', val_loss, epoch)
        self.writer.add_scalar('Dice/Val', val_dice, epoch)
        self.writer.add_scalar('LearningRate', lr, epoch)
        
        # Add training statistics
        for key, value in train_stats.items():
            if key != 'loss':  # Already logged above
                self.writer.add_scalar(f'Train/{key}', value, epoch)
        
        # Print summary
        print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_stats['dice_score']:.4f}")
        if 'kidney_ratio' in train_stats:
            print(f"       Kidney ratio: {train_stats['kidney_ratio']:.4%}")
        print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")
        print(f"Learning Rate: {lr:.6f}")
        print(f"Mean tumor ratio: {train_stats['tumor_ratio']:.4%}")