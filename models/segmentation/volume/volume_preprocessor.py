import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
import gc
import nibabel as nib
import time

logger = logging.getLogger(__name__)

def print_progress(msg: str):
    """Print with timestamp and flush immediately"""
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    print(f"[{timestamp}] {msg}", flush=True)

class VolumePreprocessor:
    """Preprocessor for full-volume segmentation data"""
    
    def __init__(self, config):
        """Initialize preprocessor with configuration"""
        self.config = config
        self.processing_dtype = np.float32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create output directory
        self.output_dir = Path(config.output_dir) / "preprocessed_volumes"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get window range from config
        self.window_range = getattr(config, 'window_range', (-1024, 1024))
        
        # Target shape for downsampling (e.g., 128,256,256)
        self.target_shape = getattr(config, 'volume_max_dim', (128, 256, 256))

    def preprocess_case(self, case_path: Path) -> Dict[str, Any]:
        """
        Preprocess a single case for volume training
        
        Args:
            case_path: Path to case directory containing imaging.nii.gz and segmentation.nii.gz
            
        Returns:
            Dictionary containing preprocessed tensors and metadata
        """
        print_progress(f"Processing case: {case_path.name}")
        
        try:
            # Load CT
            img_path = case_path / "imaging.nii.gz"
            img_obj = nib.load(str(img_path))
            spacing = img_obj.header.get_zooms()
            original_shape = img_obj.shape
            
            print_progress("  Loading CT data...")
            # Load directly as float32
            ct_data = np.asarray(img_obj.dataobj, dtype=self.processing_dtype)
            
            # Window and normalize
            print_progress("  Windowing and normalizing...")
            min_hu, max_hu = self.window_range
            ct_data = np.clip(ct_data, min_hu, max_hu)
            ct_min, ct_max = ct_data.min(), ct_data.max()
            ct_norm = (ct_data - ct_min) / (ct_max - ct_min)
            
            # Clean up original data
            del ct_data
            gc.collect()
            
            # Load segmentation
            print_progress("  Loading segmentation...")
            seg_path = case_path / "segmentation.nii.gz"
            seg_obj = nib.load(str(seg_path))
            seg_data = np.asarray(seg_obj.dataobj, dtype=self.processing_dtype)
            
            print_progress("  Creating masks...")
            # Create masks
            kidney_mask = (seg_data == 1).astype(self.processing_dtype)
            tumor_mask = (seg_data == 2).astype(self.processing_dtype)
            cyst_mask = (seg_data == 3).astype(self.processing_dtype)
            
            # Combined kidney mask includes all three
            combined_mask = ((seg_data == 1) | (seg_data == 2) | (seg_data == 3)).astype(self.processing_dtype)
            
            # Clean up segmentation data
            del seg_data
            gc.collect()
            
            print_progress("  Downsampling volumes...")
            # Stack all volumes for efficient downsampling
            vol = np.stack([ct_norm, combined_mask, tumor_mask, cyst_mask], axis=0)  # [4,D,H,W]
            vol_tensor = torch.from_numpy(vol)[None].to(self.device)  # [1,4,D,H,W]
            
            # Downsample all volumes together
            vol_ds = F.interpolate(
                vol_tensor, 
                size=self.target_shape,
                mode='trilinear',
                align_corners=False
            )  # [1,4,D_new,H_new,W_new]
            
            # Split channels back out
            ct_tensor = vol_ds[:, 0:1]      # [1,1,D,H,W]
            kidney_tensor = vol_ds[:, 1:2]   # [1,1,D,H,W]
            tumor_tensor = vol_ds[:, 2:3]    # [1,1,D,H,W]
            cyst_tensor = vol_ds[:, 3:4]     # [1,1,D,H,W]
            
            # Clean up full res data
            del vol, vol_tensor, vol_ds
            torch.cuda.empty_cache()
            gc.collect()
            
            # Move back to CPU
            ct_tensor = ct_tensor.cpu().squeeze(0)      # [1,D,H,W]
            kidney_tensor = kidney_tensor.cpu().squeeze(0)  # [1,D,H,W]
            tumor_tensor = tumor_tensor.cpu().squeeze(0)    # [1,D,H,W]
            cyst_tensor = cyst_tensor.cpu().squeeze(0)      # [1,D,H,W]
            
            # Create metadata dictionary
            metadata = {
                'case_id': case_path.name,
                'image': ct_tensor,
                'kidney': kidney_tensor,
                'tumor': tumor_tensor,
                'cyst': cyst_tensor,
                'original_shape': original_shape,
                'target_shape': self.target_shape,
                'original_spacing': spacing,
                'stats': {
                    'ct_range': (float(ct_min), float(ct_max)),
                    'window_range': self.window_range,
                    'kidney_voxels': int(kidney_tensor.sum().item()),
                    'tumor_voxels': int(tumor_tensor.sum().item()),
                    'cyst_voxels': int(cyst_tensor.sum().item())
                }
            }
            
            # Save preprocessed data
            output_path = self.output_dir / f"case_{case_path.name}.pt"
            print_progress(f"  Saving to {output_path}")
            self.save_preprocessed(metadata, output_path)
            
            print_progress(f"  Complete! Memory cleaned up.")
            return metadata
            
        except Exception as e:
            print_progress(f"Error processing {case_path.name}: {str(e)}")
            return None
            
        finally:
            # Final cleanup
            torch.cuda.empty_cache()
            gc.collect()

    def save_preprocessed(self, metadata: Dict[str, Any], output_path: Path) -> None:
        """Save preprocessed data to disk"""
        torch.save(metadata, output_path)

    def load_preprocessed(self, file_path: Path) -> Dict[str, Any]:
        """Load preprocessed data from disk"""
        return torch.load(file_path)

    def preprocess_dataset(self, data_dir: Path) -> None:
        """Preprocess entire dataset, skipping already processed cases"""
        case_folders = sorted([f for f in data_dir.iterdir() if f.is_dir()])
        print_progress(f"Found {len(case_folders)} cases in {data_dir}")
        print_progress(f"Target shape: {self.target_shape}")
        
        # Check which cases have already been processed
        processed_files = set(f.stem.replace("case_", "") for f in self.output_dir.glob("case_*.pt"))
        remaining_cases = [f for f in case_folders if f.name not in processed_files]
        
        if len(processed_files) > 0:
            print_progress(f"Found {len(processed_files)} already processed cases:")
            for case in processed_files:
                print_progress(f"  - {case}")
            print_progress(f"Remaining cases to process: {len(remaining_cases)}")
            
        if not remaining_cases:
            print_progress("All cases have already been processed!")
            return
        
        success_count = 0
        start_time = time.time()
        
        for i, case_folder in enumerate(remaining_cases, 1):
            print_progress(f"\nProcessing case {i}/{len(remaining_cases)} ({case_folder.name})")
            
            # Check again in case file was created by another process
            output_path = self.output_dir / f"case_{case_folder.name}.pt"
            if output_path.exists():
                print_progress(f"  Skipping {case_folder.name} - already processed")
                success_count += 1
                continue
                
            if self.preprocess_case(case_folder) is not None:
                success_count += 1
            
            # Show progress stats
            elapsed = time.time() - start_time
            cases_per_hour = i / (elapsed / 3600)
            remaining = len(remaining_cases) - i
            est_remaining = remaining / cases_per_hour if cases_per_hour > 0 else 0
            
            print_progress(f"\nProgress: {i}/{len(remaining_cases)} remaining cases ({i/len(remaining_cases)*100:.1f}%)")
            print_progress(f"Total processed: {len(processed_files) + i}/{len(case_folders)} cases")
            print_progress(f"Success rate: {success_count/i*100:.1f}%")
            print_progress(f"Processing speed: {cases_per_hour:.1f} cases/hour")
            print_progress(f"Estimated time remaining: {est_remaining:.1f} hours")
            print_progress("="*80)
                
        print_progress(f"\nPreprocessing complete!")
        print_progress(f"Successfully preprocessed {success_count} new cases")
        print_progress(f"Total processed cases: {len(processed_files) + success_count}/{len(case_folders)}")