import os
import nibabel as nib
from pathlib import Path
import numpy as np
from tqdm import tqdm

def analyze_tumor_masks(data_dir: str):
    """Analyze tumor masks in preprocessed volumes"""
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        return
        
    print(f"Looking in directory: {data_dir}")
    print("\nAll files in directory:")
    all_files = sorted(os.listdir(data_dir))
    for f in all_files[:10]:  # Show first 10 files
        print(f"  {f}")
    print(f"... and {len(all_files) - 10} more files")
    
    # Look for .nii.gz files directly in the preprocessed directory
    tumor_masks = []
    zero_masks = []
    pos_masks = []
    tumor_volumes = []
    
    print("\nAnalyzing tumor masks...")
    for fname in tqdm(sorted(os.listdir(data_dir))):
        # Look for tumor mask files (both segmentation and kidney masks)
        if not (fname.endswith("_seg.nii.gz") or fname.endswith("_kidney.nii.gz")):
            continue
            
        mask_path = data_dir / fname
        mask = nib.load(mask_path).get_fdata()
        print(f"\nAnalyzing {fname}")
        print(f"Shape: {mask.shape}")
        print(f"Unique values: {np.unique(mask)}")
        
        if fname.endswith("_seg.nii.gz"):  # Only count tumor segmentation masks
            tumor_volume = float(mask.sum())
            tumor_masks.append(fname)
            
            if tumor_volume == 0:
                zero_masks.append(fname)
            else:
                pos_masks.append(fname)
                tumor_volumes.append(tumor_volume)
    
    print("\nAnalysis Results:")
    print(f"Total files in directory: {len(all_files)}")
    print(f"Total tumor masks found: {len(tumor_masks)}")
    print(f"Cases with tumor: {len(pos_masks)}")
    print(f"Cases without tumor: {len(zero_masks)}")
    
    if tumor_volumes:
        print(f"\nTumor volume statistics:")
        print(f"Mean volume: {np.mean(tumor_volumes):.2f}")
        print(f"Median volume: {np.median(tumor_volumes):.2f}")
        print(f"Min volume: {np.min(tumor_volumes):.2f}")
        print(f"Max volume: {np.max(tumor_volumes):.2f}")
    
    if len(zero_masks) > 0:
        print("\nFirst few cases without tumor:")
        for fname in zero_masks[:5]:
            print(f"\n  {fname}")
            # Load and verify the mask is truly zero
            mask_path = data_dir / fname
            mask = nib.load(mask_path).get_fdata()
            print(f"  Verification:")
            print(f"    Sum: {mask.sum():.2f}")
            print(f"    Max: {mask.max():.2f}")
            print(f"    Unique values: {np.unique(mask)}")
    else:
        print("\nNo zero-tumor cases found!")

if __name__ == "__main__":
    # Get the data directory from environment variable or use default
    data_dir = os.getenv("PREPROCESSED_DIR", "/workspace/RCC_Project/preprocessed_volumes")
    analyze_tumor_masks(data_dir)