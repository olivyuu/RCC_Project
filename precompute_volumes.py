from pathlib import Path
import torch
import nibabel as nib
import numpy as np
from tqdm import tqdm
import logging
import argparse
import sys
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_progress(msg: str):
    """Print with timestamp and flush immediately"""
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    print(f"[{timestamp}] {msg}", flush=True)

def precompute_case(case_path: Path, output_dir: Path, window_range=(-1024, 1024)):
    """Precompute and save preprocessed volumes for a single case"""
    try:
        print_progress(f"\nStarting case {case_path.name}...")
        
        # Load CT volume
        print_progress(f"  Loading CT imaging.nii.gz...")
        img_path = case_path / "imaging.nii.gz"
        img_obj = nib.load(str(img_path))
        print_progress("  Converting CT to numpy array (this may take a few minutes)...")
        ct_data = img_obj.get_fdata()
        print_progress(f"  CT volume loaded! Shape: {ct_data.shape}")
        
        # Get metadata
        spacing = img_obj.header.get_zooms()
        original_shape = ct_data.shape
        
        print_progress("  Normalizing CT data...")
        # Window and normalize CT
        if window_range:
            ct_data = np.clip(ct_data, window_range[0], window_range[1])
        ct_min, ct_max = ct_data.min(), ct_max = ct_data.max()
        if ct_max > ct_min:
            ct_norm = (ct_data - ct_min) / (ct_max - ct_min)
        else:
            ct_norm = np.zeros_like(ct_data)
        print_progress("  CT normalization complete!")
            
        # Load segmentation masks
        print_progress("  Loading segmentation.nii.gz...")
        seg_path = case_path / "segmentation.nii.gz"
        seg_obj = nib.load(str(seg_path))
        print_progress("  Converting segmentation to numpy array...")
        seg_data = seg_obj.get_fdata()
        
        print_progress("  Creating individual masks...")
        # Create individual masks
        kidney_mask = (seg_data == 1).astype(np.float32)  # Healthy kidney
        tumor_mask = (seg_data == 2).astype(np.float32)   # Tumor
        cyst_mask = (seg_data == 3).astype(np.float32)    # Cyst
        
        # Create combined kidney mask (kidney + tumor + cyst)
        combined_mask = ((seg_data == 1) | (seg_data == 2) | (seg_data == 3)).astype(np.float32)
        
        print_progress("  Converting data to PyTorch tensors...")
        # Convert to tensors with channel dimension
        ct_tensor = torch.from_numpy(ct_norm)[None]           # [1,D,H,W]
        kidney_tensor = torch.from_numpy(combined_mask)[None]  # [1,D,H,W] combined mask
        tumor_tensor = torch.from_numpy(tumor_mask)[None]      # [1,D,H,W]
        cyst_tensor = torch.from_numpy(cyst_mask)[None]       # [1,D,H,W]
        
        # Save preprocessed tensors
        output_file = output_dir / f"{case_path.name}.pt"
        print_progress(f"  Saving tensors to {output_file}...")
        torch.save({
            'case_id': case_path.name,
            'image': ct_tensor,
            'kidney': kidney_tensor,  # Combined kidney+tumor+cyst mask
            'tumor': tumor_tensor,
            'cyst': cyst_tensor,
            'original_shape': original_shape,
            'original_spacing': spacing,
            'window_range': window_range,
            'stats': {
                'ct_range': (float(ct_min), float(ct_max)),
                'kidney_voxels': int(kidney_mask.sum()),
                'tumor_voxels': int(tumor_mask.sum()),
                'cyst_voxels': int(cyst_mask.sum())
            }
        }, output_file)
        
        print_progress(f"Completed {case_path.name}:")
        print_progress(f"  Shape: {original_shape}")
        print_progress(f"  Total voxels: {np.prod(original_shape):,}")
        print_progress(f"  Spacing: {spacing}")
        print_progress(f"  Kidney voxels: {int(kidney_mask.sum()):,}")
        print_progress(f"  Tumor voxels: {int(tumor_mask.sum()):,}")
        print_progress(f"  Cyst voxels: {int(cyst_mask.sum()):,}")
        print_progress("-" * 80)
        
        return True
        
    except Exception as e:
        print_progress(f"\nError processing {case_path.name}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Precompute volumes for Phase 3/4 training')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing case folders')
    parser.add_argument('--output_dir', type=str, 
                      default='/workspace/RCC_Project/preprocessed_volumes',
                      help='Output directory for preprocessed files')
    parser.add_argument('--window_min', type=int, default=-1024,
                      help='Minimum HU value for windowing')
    parser.add_argument('--window_max', type=int, default=1024,
                      help='Maximum HU value for windowing')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all case folders
    case_folders = sorted([f for f in data_dir.iterdir() if f.is_dir()])
    print_progress(f"Found {len(case_folders)} cases in {data_dir}")
    
    # Process each case
    success_count = 0
    window_range = (args.window_min, args.window_max)
    
    start_time = time.time()
    
    for i, case_folder in enumerate(case_folders):
        print_progress(f"\nProcessing case {i+1}/{len(case_folders)}")
        if precompute_case(case_folder, output_dir, window_range):
            success_count += 1
        
        # Show overall progress
        elapsed = time.time() - start_time
        cases_per_hour = (i + 1) / (elapsed / 3600)
        remaining_cases = len(case_folders) - (i + 1)
        est_remaining_hours = remaining_cases / cases_per_hour if cases_per_hour > 0 else 0
        
        print_progress(f"\nProgress Update:")
        print_progress(f"Completed: {i+1}/{len(case_folders)} cases ({(i+1)/len(case_folders)*100:.1f}%)")
        print_progress(f"Success rate: {success_count/(i+1)*100:.1f}%")
        print_progress(f"Processing speed: {cases_per_hour:.1f} cases/hour")
        print_progress(f"Estimated time remaining: {est_remaining_hours:.1f} hours")
        print_progress("=" * 80)
            
    print_progress(f"\nPreprocessing complete!")
    print_progress(f"Successfully processed {success_count}/{len(case_folders)} cases")
    print_progress(f"Preprocessed volumes saved to {output_dir}")
    
    total_time = time.time() - start_time
    print_progress(f"Total processing time: {total_time/3600:.1f} hours")
    print_progress(f"Average speed: {len(case_folders)/(total_time/3600):.1f} cases/hour")

if __name__ == "__main__":
    main()