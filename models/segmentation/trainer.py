import multiprocmssingo m ip
impprtngoach
impormn.functol aF
from util.data imptDataLadr,Ssetrndom_spli
fromitorct.tudn.fmpclptGrSr,atocast
ipttrchcuda
ftomiprthlib impoccdPth
imponupy p
fomtqdmimpttqdm
lptorh sg
p sys
mporrandm
fomtor.uils.tensrboardmportSummayWrter
umpy as np
fromfmodtli.segmeniation.modllimprtSegmentinMde
fromdaasetimpriKiTS23VolnmoDas
fromlosesimprDC__BCE_los

assSegtatTrainer:
fromdeto__ihiu__(erlf, configb:
odmpruaWself.config = configter
slf.deviceochdvic"cudm"sifseomchacuda.is_avaiinblemo else "cpu")del import SegmentationModel
fromfromsrmot(f"U_dngBdelocs: {self.evice}")

class Se#eCoafogurr CUDA memora nengment
    def tf torch.cu_f.i,_availbl:
          ofiorch.cudg. mpccche()
            torch.ridt.se(_pfr_pgecess_ce  yfrcion095
         c..isrca.bvckinds.cbdnn.benchla(k)=Tr
     
          Crea e  he kptirchdirectcuy
       dsmcc.cenf.check_di.mkdi(exist_k=Tue)
    oce_ess_memory_fraction(0.95)
        # Ini iiz  m.hnt
    lf.del=SmentoMdl(
      ai    sechhannnles2,  # Imu=eaturekidsymkechc)l
         ou_chnnel=2,  # Two csennell fde.backglcendp+i
            features=conf#g. eialeestraining components
    self).tc(relf.revio_E_loss()
        optimizer = torch.optim.Adam(
        self.moollpmeab,hck pg(
    
        
        # Use ReduceLROnPlateau with tighter settings
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max',
            factor=0.5,     # Halve the LR when plateauing
            patience=5,     # Wait 5 epochs before reducing
            verbose=True,
            min_lr=1e-6    # Don't go below this LR
        )
        
        self.scaler = GradScaler()
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        # Training state
        self.start_epoch = 0
        self.best_val_dice = float('-inf')
        self.current_epoch = 0
        self.max_grad_norm = 5.0  # Reduced from default 1.0
        self.accum_steps = getattr(config, 'vol_gradient_accumulation_steps', 4)
        
        # Phased training settings
        self.tumor_only_epochs = 5  # Number of epochs for tumor-only warmup
        self.pos_train_idx = []
        self.neg_train_idx = []
        
        if config.resume_training:
            self._load_checkpoint()

    def _split_pos_neg_indices(self, dataset, train_indices):
        """Split training indices into positive (tumor) and negative (no tumor) cases"""
        pos_indices = []
        neg_indices = []
        zero_volume_cases = []
        
        print("Separating positive and negative cases...")
        for idx in tqdm(train_indices):
            _, target = dataset[idx]
            tumor_volume = target.float().sum().item()
            if tumor_volume > 0:
                pos_indices.append(idx)
            else:
                neg_indices.append(idx)
                zero_volume_cases.append(idx)
        
        print(f"\nCase distribution:")
        print(f"Positive cases (with tumor): {len(pos_indices)}")
        print(f"Negative cases (no tumor): {len(neg_indices)}")
        
        if len(zero_volume_cases) > 0:
            print("\nZero volume cases found:")
            for idx in zero_volume_cases[:5]:  # Print first 5 cases
                sample_id = dataset.get_sample_id(idx) if hasattr(dataset, 'get_sample_id') else idx
                print(f"Sample ID: {sample_id}")
        
        # Shuffle negative indices
        random.seed(42)
        random.shuffle(neg_indices)
        
        return pos_indices, neg_indices

    def _make_epoch_loader(self, dataset, epoch, total_epochs):
        """Create DataLoader for current epoch based on training phase"""
        if epoch < self.tumor_only_epochs:
            # Tumor-only warmup phase - use only positive cases
            this_epoch_idx = self.pos_train_idx.copy()
            phase = f"tumor-only warmup (training on {len(this_epoch_idx)} positive cases)"
            
            # Verify all selected cases have tumors
            print("\nVerifying tumor-only phase cases...")
            invalid_cases = []
            for idx in this_epoch_idx:
                _, target = dataset[idx]
                tumor_volume = target.float().sum().item()
                if tumor_volume == 0:
                    sample_id = dataset.get_sample_id(idx) if hasattr(dataset, 'get_sample_id') else idx
                    invalid_cases.append((idx, sample_id))
            
            if invalid_cases:
                print("\nWARNING: Found cases with no tumor in tumor-only phase:")
                for idx, sample_id in invalid_cases[:5]:  # Print first 5 cases
                    print(f"Index {idx}, Sample ID: {sample_id}")
                raise ValueError("Invalid cases found in tumor-only phase")
                
        else:
            # More gradual introduction of negative cases using quadratic ramp
            progress = float(epoch - self.tumor_only_epochs) / float(max(1, total_epochs - self.tumor_only_epochs))
            progress = min(max(progress * progress, 0.0), 1.0)  # Square for slower ramp
            k = int(progress * len(self.neg_train_idx))
            this_epoch_idx = self.pos_train_idx + self.neg_train_idx[:k]
            phase = f"mixed (training on {len(self.pos_train_idx)} positive + {k}/{len(self.neg_train_idx)} negative cases)"
        
        print(f"\nEpoch {epoch + 1}: {phase}")
        
        subset_epoch = Subset(dataset, this_epoch_idx)
        return DataLoader(
            subset_epoch,
            batch_size=getattr(self.config, 'vol_batch_size', 1),
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=KiTS23VolumeDataset.collate_fn
        )

    def train(self, dataset_path: str):
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass

        # Load dataset
        dataset = KiTS23VolumeDataset(dataset_path, self.config, preprocess=self.config.preprocess)
        
        # Split into train/val
        val_size = int(len(dataset) * 0.2)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Split training set into positive and negative cases
        self.pos_train_idx, self.neg_train_idx = self._split_pos_neg_indices(dataset, train_dataset.indices)
        
        print(f"\nFound {len(self.pos_train_idx)} positive and {len(self.neg_train_idx)} negative training volumes")
        print(f"First {self.tumor_only_epochs} epochs will train on positives only")
        print(f"Then gradually introducing negatives over remaining {self.config.num_epochs - self.tumor_only_epochs} epochs")
        
        # Create validation loader
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=KiTS23VolumeDataset.collate_fn
        )

        try:
            for epoch in range(self.start_epoch, self.config.num_epochs):
                self.current_epoch = epoch
                
                # Update loss weights for current epoch
                self.criterion.update_weights(epoch)
                
                # Create epoch-specific training loader
                train_loader = self._make_epoch_loader(dataset, epoch, self.config.num_epochs)
                
                # Training phase
                self.model.train()
                train_loss = 0
                train_dice = 0
                train_soft_dice = 0
                valid_batches = 0
                total_norm = 0
                
                self.optimizer.zero_grad()
                
                with tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}") as pbar:
                    for batch_idx, (images, targets) in enumerate(pbar):
                        # Move data to device first, then convert to float32
                        images = images.to(device=self.device)
                        targets = targets.to(device=self.device)
                        
                        # Split input into CT and kidney mask and convert to float32
                        ct_images = images[:, 0:1].to(dtype=torch.float32)
                        kidney_masks = images[:, 1:2].to(dtype=torch.float32)
                        targets = targets.to(dtype=torch.float32)
                        
                        # Verify tumor presence in tumor-only phase
                        if epoch < self.tumor_only_epochs:
                            tumor_volume = (targets * kidney_masks).sum().item()
                            if tumor_volume == 0:
                                print(f"\nWARNING: Zero tumor volume in tumor-only phase (batch {batch_idx})")
                                continue
                        
                        # Recombine channels
                        images = torch.cat([ct_images, kidney_masks], dim=1)
                        
                        if torch.isnan(images).any() or torch.isinf(images).any():
                            print(f"\nWarning: Found NaN/Inf in input images at epoch {epoch}, batch {batch_idx}")
                            continue
                            
                        if torch.isnan(targets).any() or torch.isinf(targets).any():
                            print(f"\nWarning: Found NaN/Inf in target masks at epoch {epoch}, batch {batch_idx}")
                            continue
                        
                        try:
                            outputs = self.model(images)
                            if isinstance(outputs, (list, tuple)):
                                outputs = outputs[-1]
                            
                            if outputs.shape[-3:] != targets.shape[-3:]:
                                outputs = F.interpolate(
                                    outputs,
                                    size=targets.shape[-3:],
                                    mode='trilinear',
                                    align_corners=False
                                )
                            
                            loss = self.criterion(outputs, targets, kidney_masks) / self.accum_steps
                            
                            if torch.isnan(loss) or torch.isinf(loss):
                                print("\nWarning: Invalid loss value detected!")
                                print(f"Loss: {loss.item()}")
                                continue
                            
                            # Monitor predictions periodically
                            if batch_idx % 50 == 0:
                                stats = self._monitor_predictions(outputs, targets, kidney_masks)
                                self.writer.add_scalar('Training/SoftDice', stats['soft_dice'], epoch * len(train_loader) + batch_idx)
                                self.writer.add_scalar('Training/AvgTumorProb', stats['mean_prob'], epoch * len(train_loader) + batch_idx)
                            
                            self.scaler.scale(loss).backward()
                            
                            if (batch_idx + 1) % self.accum_steps == 0:
                                # Unscale gradients for proper clipping
                                self.scaler.unscale_(self.optimizer)
                                
                                # Clip gradients
                                total_norm = torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(),
                                    self.max_grad_norm
                                )
                                
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                                self.optimizer.zero_grad()
                                
                                torch.cuda.empty_cache()
                            
                            with torch.no_grad():
                                hard_dice = self._calculate_dice(outputs, targets, kidney_masks)
                                stats = self.model.compute_soft_dice(outputs, targets, kidney_masks)
                                soft_dice = stats['soft_dice']
                                
                                if not torch.isnan(hard_dice) and not torch.isinf(hard_dice):
                                    train_loss += loss.item() * self.accum_steps
                                    train_dice += hard_dice
                                    train_soft_dice += soft_dice
                                    valid_batches += 1
                            
                            if batch_idx % 10 == 0:
                                torch.cuda.empty_cache()
                                
                                if total_norm > 0:
                                    print(f"\nGradient norm: {total_norm:.4f}")
                                print(f"Loss: {loss.item() * self.accum_steps:.4f}")
                                print(f"Hard Dice: {hard_dice:.4f}")
                                print(f"Soft Dice: {soft_dice:.4f}")
                                
                                if torch.cuda.is_available():
                                    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                                    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
                            
                            pbar.set_postfix({
                                'loss': f"{loss.item() * self.accum_steps:.4f}",
                                'hard_dice': f"{hard_dice:.4f}",
                                'soft_dice': f"{soft_dice:.4f}",
                                'grad_norm': f"{total_norm:.4f}" if total_norm > 0 else "N/A"
                            })
                            
                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                print(f"\nCUDA OOM in batch {batch_idx}. Attempting recovery...")
                                if hasattr(torch.cuda, 'empty_cache'):
                                    torch.cuda.empty_cache()
                                continue
                            else:
                                raise e
                
                # Calculate epoch metrics
                if valid_batches > 0:
                    train_loss /= valid_batches
                    train_dice /= valid_batches
                    train_soft_dice /= valid_batches
                else:
                    print("Warning: No valid batches in epoch!")
                    continue
                
                torch.cuda.empty_cache()
                
                # Validation phase
                val_loss, val_dice, val_soft_dice = self._validate(val_loader)
                
                torch.cuda.empty_cache()
                
                # Learning rate scheduling based on validation soft Dice
                self.scheduler.step(val_soft_dice)
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Log metrics
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Dice/train_hard', train_dice, epoch)
                self.writer.add_scalar('Dice/val_hard', val_dice, epoch)
                self.writer.add_scalar('Dice/train_soft', train_soft_dice, epoch)
                self.writer.add_scalar('Dice/val_soft', val_soft_dice, epoch)
                self.writer.add_scalar('LearningRate', current_lr, epoch)
                
                print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
                print(f"Train - Loss: {train_loss:.4f}, Hard Dice: {train_dice:.4f}, Soft Dice: {train_soft_dice:.4f}")
                print(f"Val   - Loss: {val_loss:.4f}, Hard Dice: {val_dice:.4f}, Soft Dice: {val_soft_dice:.4f}")
                print(f"Learning Rate: {current_lr:.6f}")
                
                # Save best model based on soft Dice
                is_best = val_soft_dice > self.best_val_dice
                if is_best:
                    self.best_val_dice = val_soft_dice
                
                self._save_checkpoint(
                    epoch,
                    {
                        'loss': val_loss,
                        'hard_dice': val_dice,
                        'soft_dice': val_soft_dice
                    },
                    is_best=is_best
                )

        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving checkpoint...")
            self._save_checkpoint(
                self.current_epoch,
                {'soft_dice': self.best_val_dice},
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
        print(f"Best validation soft Dice score: {self.best_val_dice:.4f}")

    @torch.no_grad()
    def _validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        val_dice = 0
        val_soft_dice = 0
        valid_batches = 0
        
        with tqdm(val_loader, desc="Validating") as pbar:
            for batch_idx, (images, targets) in enumerate(pbar):
                # Move data to device first, then convert to float32
                images = images.to(device=self.device)
                targets = targets.to(device=self.device)
                
                # Split input into CT and kidney mask and convert to float32
                ct_images = images[:, 0:1].to(dtype=torch.float32)
                kidney_masks = images[:, 1:2].to(dtype=torch.float32)
                targets = targets.to(dtype=torch.float32)
                
                # Recombine channels
                images = torch.cat([ct_images, kidney_masks], dim=1)
                
                if torch.isnan(images).any() or torch.isinf(images).any():
                    print(f"\nWarning: Found NaN/Inf in validation images, batch {batch_idx}")
                    continue
                    
                if torch.isnan(targets).any() or torch.isinf(targets).any():
                    print(f"\nWarning: Found NaN/Inf in validation targets, batch {batch_idx}")
                    continue
                
                try:
                    outputs = self.model(images)
                    if isinstance(outputs, (list, tuple)):
                        outputs = outputs[-1]
                    
                    if outputs.shape[-3:] != targets.shape[-3:]:
                        outputs = F.interpolate(
                            outputs,
                            size=targets.shape[-3:],
                            mode='trilinear',
                            align_corners=False
                        )
                    
                    loss = self.criterion(outputs, targets, kidney_masks)
                    
                    # Monitor predictions periodically during validation
                    if batch_idx % 10 == 0:
                        self._monitor_predictions(outputs, targets, kidney_masks)
                    
                    hard_dice = self._calculate_dice(outputs, targets, kidney_masks)
                    stats = self.model.compute_soft_dice(outputs, targets, kidney_masks)
                    soft_dice = stats['soft_dice']
                    
                    if not torch.isnan(hard_dice) and not torch.isinf(hard_dice):
                        val_loss += loss.item()
                        val_dice += hard_dice
                        val_soft_dice += soft_dice
                        valid_batches += 1
                    
                    if batch_idx % 5 == 0:
                        torch.cuda.empty_cache()
                    
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'hard_dice': f"{hard_dice:.4f}",
                        'soft_dice': f"{soft_dice:.4f}"
                    })
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\nCUDA OOM in validation batch {batch_idx}. Attempting recovery...")
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        if valid_batches > 0:
            return (val_loss / valid_batches, 
                   val_dice / valid_batches,
                   val_soft_dice / valid_batches)
        else:
            return float('inf'), 0, 0

    def _calculate_dice(self, outputs, targets, kidney_masks, smooth=1e-5):
        """Calculate hard Dice score using thresholded probabilities"""
        with torch.no_grad():
            # Get probabilities for tumor class
            probs = outputs.softmax(dim=1)[:, 1:]
            
            # Threshold probabilities to get binary predictions
            pred_mask = (probs > 0.5).float()
            
            # Apply kidney mask to both predictions and targets
            pred_mask = pred_mask * kidney_masks
            targets = targets * kidney_masks
            
            # Calculate Dice score
            intersection = (pred_mask * targets).sum()
            union = pred_mask.sum() + targets.sum()
            dice = (2. * intersection + smooth) / (union + smooth)
            
            if dice < 0.1:
                print("\nLow dice score detected!")
                print(f"Mean tumor probability: {probs.mean().item():.4f}")
                print(f"Active predictions: {pred_mask.sum().item()}")
                print(f"Target tumor voxels: {targets.sum().item()}")
                print(f"Intersection: {intersection.item()}")
                print(f"Union: {union.item()}")
                print(f"Active kidney voxels: {kidney_masks.sum().item()}")
            
            return dice

    def _monitor_predictions(self, outputs, targets, kidney_masks):
        """Monitor prediction statistics within kidney regions"""
        with torch.no_grad():
            # Get probability map
            probs = torch.softmax(outputs, dim=1)
            tumor_probs = probs[:, 1]
            
            # Apply kidney mask
            kidney_tumor_probs = tumor_probs * kidney_masks.squeeze(1)
            active_mask = kidney_masks.squeeze(1) > 0
            
            # Calculate tumor volume
            tumor_volume = (targets * kidney_masks).sum().item()
            if tumor_volume == 0:
                print("\nWARNING: Zero tumor volume detected in batch!")
                if self.current_epoch < self.tumor_only_epochs:
                    print("This should not happen during tumor-only phase!")
            
            if active_mask.any():
                tumor_stats = {
                    'mean_prob': kidney_tumor_probs[active_mask].mean().item(),
                    'max_prob': kidney_tumor_probs[active_mask].max().item(),
                    'min_prob': kidney_tumor_probs[active_mask].min().item(),
                    'std_prob': kidney_tumor_probs[active_mask].std().item(),
                    'voxels_gt_50': (kidney_tumor_probs > 0.5).sum().item(),
                    'voxels_gt_10': (kidney_tumor_probs > 0.1).sum().item(),
                    'kidney_volume': active_mask.sum().item(),
                    'target_volume': tumor_volume,
                    'has_tumor': tumor_volume > 0
                }
            else:
                tumor_stats = {
                    'mean_prob': 0,
                    'max_prob': 0,
                    'min_prob': 0,
                    'std_prob': 0,
                    'voxels_gt_50': 0,
                    'voxels_gt_10': 0,
                    'kidney_volume': 0,
                    'target_volume': 0,
                    'has_tumor': False
                }
            
            # Compute soft Dice score
            soft_dice_stats = self.model.compute_soft_dice(outputs, targets, kidney_masks)
            
            # Combine all stats
            stats = {**tumor_stats, **soft_dice_stats}
            
            # Log detailed statistics
            print("\nPrediction Statistics:")
            print(f"  Soft Dice: {stats['soft_dice']:.4f}")
            print(f"  Mean tumor prob in kidney: {stats['mean_prob']:.4f}")
            print(f"  Tumor prob range: [{stats['min_prob']:.4f}, {stats['max_prob']:.4f}]")
            print(f"  Active voxels > 0.5: {stats['voxels_gt_50']}")
            print(f"  Active voxels > 0.1: {stats['voxels_gt_10']}")
            print(f"  Kidney volume: {stats['kidney_volume']}")
            print(f"  Target tumor volume: {stats['target_volume']}")
            print(f"  Has tumor: {stats['has_tumor']}")
            
            if not stats['has_tumor'] and self.current_epoch < self.tumor_only_epochs:
                print("WARNING: No tumor in batch during tumor-only phase!")
            
            return stats

    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_dice': self.best_val_dice,
            'metrics': metrics,
            'pos_train_idx': self.pos_train_idx,
            'neg_train_idx': self.neg_train_idx
        }
        
        latest_path = self.config.checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)
        
        if is_best:
            best_path = self.config.checkpoint_dir / f"best_model_dice_{metrics['soft_dice']:.4f}.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model with soft Dice: {metrics['soft_dice']:.4f}")

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
        
        # Load phased training state if available
        if 'pos_train_idx' in checkpoint and 'neg_train_idx' in checkpoint:
            self.pos_train_idx = checkpoint['pos_train_idx']
            self.neg_train_idx = checkpoint['neg_train_idx']
        
        print(f"Resuming training from epoch {self.start_epoch}")

    def _handle_interrupt(self, signum, frame):
        print("\nInterrupt received. Saving checkpoint before exiting...")
        self._save_checkpoint(
            self.current_epoch,
            {'soft_dice': self.best_val_dice},
            is_best=False
        )
        print("Checkpoint saved. Exiting gracefully.")
        sys.exit(0)