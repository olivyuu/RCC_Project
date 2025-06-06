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

from models.segmentation.model import SegmentationModel
from dataset_patch import KiTS23PatchDataset
from losses_patch import WeightedDiceBCELoss
from dataset_volume import KiTS23VolumeDataset

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
            out_channels=2,  # Two channels for background/tumor
            features=config.features
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
        
        if config.resume_training:
            self._load_checkpoint()

    def train(self, dataset_path: str):
        """Main training loop"""
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass

        # Create training dataset
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
            tumor_only_prob=self.tumor_only_prob,  # Use instance variable
            debug=self.config.debug
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
            persistent_workers=True
        )

        print(f"\nTraining configuration:")
        print(f"Patch size: {self.patch_size}")
        print(f"Batch size: {self.batch_size}")
        print(f"Tumor sampling probability: {self.tumor_only_prob}")
        print(f"Training patches per epoch: {len(train_loader)}")
        print(f"Validation volumes: {len(val_dataset)}")

        try:
            # Register signal handlers for graceful interruption
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
                        'dice': val_dice
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
            import traceback
            traceback.print_exc()
            raise
        
        self.writer.close()
        print("\nTraining completed!")
        print(f"Best validation Dice score: {self.best_val_dice:.4f}")

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

# Rest of the methods (_train_epoch, _validate, etc.) remain unchanged...