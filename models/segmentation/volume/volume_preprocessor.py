import torch
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
            
            print_progress("  Converting to tensors...")
            # Convert to tensors
            ct_tensor = torch.from_numpy(ct_norm)[None]           # [1,D,H,W]
            kidney_tensor = torch.from_numpy(combined_mask)[None]  # [1,D,H,W]
            tumor_tensor = torch.from_numpy(tumor_mask)[None]      # [1,D,H,W]
            cyst_tensor = torch.from_numpy(cyst_mask)[None]       # [1,D,H,W]
            
            # Clean up numpy arrays
            del ct_norm, combined_mask, tumor_mask, cyst_mask
            gc.collect()
            
            # Create metadata dictionary
            metadata = {
                'case_id': case_path.name,
                'image': ct_tensor,
                'kidney': kidney_tensor,
                'tumor': tumor_tensor,
                'cyst': cyst_tensor,
                'original_shape': original_shape,
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
        """Preprocess entire dataset"""
        case_folders = sorted([f for f in data_dir.iterdir() if f.is_dir()])
        print_progress(f"Found {len(case_folders)} cases in {data_dir}")
        
        success_count = 0
        start_time = time.time()
        
        for i, case_folder in enumerate(case_folders, 1):
            print_progress(f"\nProcessing case {i}/{len(case_folders)}")
            if self.preprocess_case(case_folder) is not None:
                success_count += 1
            
            # Show progress stats
            elapsed = time.time() - start_time
            cases_per_hour = i / (elapsed / 3600)
            remaining = len(case_folders) - i
            est_remaining = remaining / cases_per_hour if cases_per_hour > 0 else 0
            
            print_progress(f"\nProgress: {i}/{len(case_folders)} cases ({i/len(case_folders)*100:.1f}%)")
            print_progress(f"Success rate: {success_count/i*100:.1f}%")
            print_progress(f"Processing speed: {cases_per_hour:.1f} cases/hour")
            print_progress(f"Estimated time remaining: {est_remaining:.1f} hours")
            print_progress("="*80)
                
        print_progress(f"\nPreprocessing complete!")
        print_progress(f"Successfully preprocessed {success_count}/{len(case_folders)} cases")