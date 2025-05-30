import torch
from torch.utils.data import Dataset
from pathlib import Path
from preprocessing.preprocessor import KiTS23Preprocessor
from augmentations import KiTS23Augmenter
from typing import List, Tuple
import gc
import numpy as np
from contextlib import nullcontext

from tools.debug_utils import DebugLogger, validate_volume_shapes, debug_data_sample, gpu_memory_check

class KiTS23VolumeDataset(Dataset):
    """Dataset class for handling full volume data in KiTS23."""
    
    def __init__(self, root_dir: str, config, preprocessor: KiTS23Preprocessor = None,
                 train: bool = True, preprocess: bool = True, debug: bool = False):
        """
        Initialize the volume dataset.
        
        Args:
            root_dir (str): Path to the raw dataset directory
            config: Configuration object
            preprocessor (KiTS23Preprocessor, optional): Pre-initialized preprocessor
            train (bool): Whether this is training set
            preprocess (bool): Whether to preprocess volumes
        """
        self.root_dir = Path(root_dir)
        self.config = config
        self.preprocessor = preprocessor or KiTS23Preprocessor(config)
        self.augmenter = KiTS23Augmenter(config) if train else None
        
        # Define path for preprocessed volumes
        self.preprocessed_dir = Path(config.preprocessed_volumes_dir)
        self.preprocessed_dir.mkdir(exist_ok=True)
        
        # Setup debug logging if requested
        self.debug = debug
        if debug:
            self.debug_logger = DebugLogger(self.preprocessed_dir, debug=True)
        else:
            self.debug_logger = None
            
        # Store case paths for loading
        self.case_paths: List[Path] = []
        
        if preprocess:
            print(f"Preprocessing volumes from {self.root_dir}")
            all_case_paths = sorted([d for d in self.root_dir.iterdir() if d.is_dir() and d.name.startswith('case_')])
            
            for case_path in all_case_paths:
                output_file = self.preprocessed_dir / f"{case_path.name}_volume.pt"
                
                if output_file.exists():
                    try:
                        # Quick validation of saved volume
                        data = torch.load(output_file)
                        if isinstance(data, tuple) and len(data) == 3:  # Now expecting 3 elements (image, kidney_mask, tumor_mask)
                            print(f"Found valid preprocessed volume: {output_file}")
                            self.case_paths.append(case_path)
                            continue
                    except Exception as e:
                        print(f"Error validating {output_file}: {e}")
                
                print(f"Processing case {case_path.name}")
                try:
                    # Load and preprocess volume
                    volume_data = self.preprocessor.preprocess_volume(case_path)
                    if volume_data[0] is not None:  # Check if volume was successfully processed
                        torch.save(volume_data, output_file)
                        self.case_paths.append(case_path)
                except Exception as e:
                    print(f"Error processing {case_path.name}: {e}")
                    continue
                
                # Clear memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            print(f"Preprocessing complete. Total valid cases: {len(self.case_paths)}")
        
        else:  # Load existing preprocessed volumes
            print(f"Loading preprocessed volumes from {self.preprocessed_dir}")
            # If no preprocessed files exist, preprocess anyway
            all_volume_files = sorted(self.preprocessed_dir.glob("case_*_volume.pt"))
            if not all_volume_files:
                print("No preprocessed volumes found, forcing preprocessing")
                self.case_paths = sorted([d for d in self.root_dir.iterdir() if d.is_dir() and d.name.startswith('case_')])
                preprocess = True
            else:
                self.case_paths = [Path(str(f).replace('_volume.pt', '')) for f in all_volume_files]
    
    def __len__(self) -> int:
        return len(self.case_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        case_path = self.case_paths[idx]
        volume_file = self.preprocessed_dir / f"{case_path.name}_volume.pt"
        
        if self.debug_logger:
            self.debug_logger.log_memory(f"Before loading {case_path.name}")
        
        try:
            with gpu_memory_check() if self.debug_logger else nullcontext():
                # Load image, kidney mask and tumor mask
                image, kidney_mask, tumor_mask = torch.load(volume_file)
                
                # Remove any extra dimensions
                image = image.squeeze()
                kidney_mask = kidney_mask.squeeze()
                tumor_mask = tumor_mask.squeeze()
                
                if self.debug_logger:
                    self.debug_logger.log_shapes("Loaded volume", image=image, kidney_mask=kidney_mask, tumor_mask=tumor_mask)
                    self.debug_logger.log_stats("Volume stats", image=image, kidney_mask=kidney_mask, tumor_mask=tumor_mask)
                    
                    # Save detailed debug info for first batch
                    if idx == 0:
                        debug_data_sample(
                            (image, kidney_mask, tumor_mask),
                            self.preprocessed_dir / "debug_samples"
                        )
            
            # Ensure volume size doesn't exceed max dimensions
            if any(s > m for s, m in zip(image.shape[-3:], self.config.vol_max_dim)):
                if self.debug_logger:
                    self.debug_logger.log_memory("Before resizing")
                
                image = self._resize_volume(image, self.config.vol_max_dim)
                kidney_mask = self._resize_volume(kidney_mask, self.config.vol_max_dim, mode='nearest')
                tumor_mask = self._resize_volume(tumor_mask, self.config.vol_max_dim, mode='nearest')
                
                if self.debug_logger:
                    self.debug_logger.log_shapes("After resize", image=image, kidney_mask=kidney_mask, tumor_mask=tumor_mask)
            
            # Add channel dimension if needed
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            if len(kidney_mask.shape) == 3:
                kidney_mask = kidney_mask.unsqueeze(0)
            if len(tumor_mask.shape) == 3:
                tumor_mask = tumor_mask.unsqueeze(0)
            
            # Concatenate image and kidney mask as input channels
            model_input = torch.cat([image, kidney_mask], dim=0)
            
            # Apply augmentations if in training mode
            if self.augmenter is not None:
                model_input, tumor_mask = self.augmenter(model_input, tumor_mask)
            
            return model_input, tumor_mask
            
        except Exception as e:
            print(f"Error loading volume {volume_file}: {e}")
            # Return a dummy volume in case of error
            dummy_shape = (2,) + tuple(self.config.vol_max_dim)  # 2 channels for input
            return torch.zeros(dummy_shape), torch.zeros((1,) + tuple(self.config.vol_max_dim))
    
    @staticmethod
    def _resize_volume(volume: torch.Tensor, target_size: Tuple[int, int, int], mode='trilinear') -> torch.Tensor:
        """Resize a volume to target dimensions."""
        if len(volume.shape) == 3:
            volume = volume.unsqueeze(0).unsqueeze(0)
        elif len(volume.shape) == 4:
            volume = volume.unsqueeze(0)
            
        resized = torch.nn.functional.interpolate(
            volume,
            size=target_size,
            mode=mode,
            align_corners=False if mode == 'trilinear' else None
        )
        
        return resized.squeeze(0)

    @staticmethod
    def collate_fn(batch):
        """Custom collate function to handle variable sized volumes."""
        inputs, masks = zip(*batch)
        
        # Find maximum dimensions in the batch
        max_shape = [max(s) for s in zip(*[img.shape[-3:] for img in inputs])]
        
        # Pad volumes to same size
        padded_inputs = []
        padded_masks = []
        
        for inp, msk in zip(inputs, masks):
            # Calculate padding
            pad_size = [m - s for m, s in zip(max_shape, inp.shape[-3:])]
            pad = []
            for p in reversed(pad_size):  # Reverse for pytorch padding format
                pad.extend([0, p])
                
            # Pad volumes
            padded_inputs.append(torch.nn.functional.pad(inp, pad))
            padded_masks.append(torch.nn.functional.pad(msk, pad))
        
        # Stack padded volumes
        stacked_inputs = torch.stack(padded_inputs)
        stacked_masks = torch.stack(padded_masks)
        
        return stacked_inputs, stacked_masks