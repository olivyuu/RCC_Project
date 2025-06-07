from pathlib import Path
import torch
import nibabel as nib
import numpy as np
from tqdm import tqdm
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def precompute_case(case_path: Path, output_dir: Path, window_range=(-1024, 1024)):
    """Precompute and save preprocessed volumes for a single case"""
    try:
        # Load CT volume
        img_path = case_path / "imaging.nii.gz"
        img_obj = nib.load(str(img_path))
        ct_data = img_obj.get_fdata()
        
        # Get metadata
        spacing = img_obj.header.get_zooms()
        original_shape = ct_data.shape
        
        # Window and normalize CT
        if window_range:
            ct_data = np.clip(ct_data, window_range[0], window_range[1])
        ct_min, ct_max = ct_data.min(), ct_data.max()
        if ct_max > ct_min:
            ct_norm = (ct_data - ct_min) / (ct_max - ct_min)
        else:
            ct_norm = np.zeros_like(ct_data)
            
        # Load segmentation masks
        seg_path = case_path / "segmentation.nii.gz"
        seg_obj = nib.load(str(seg_path))
        seg_data = seg_obj.get_fdata()
        
        # Create individual masks
        kidney_mask = (seg_data == 1).astype(np.float32)  # Healthy kidney
        tumor_mask = (seg_data == 2).astype(np.float32)   # Tumor
        cyst_mask = (seg_data == 3).astype(np.float32)    # Cyst
        
        # Create combined kidney mask (kidney + tumor + cyst)
        combined_mask = ((seg_data == 1) | (seg_data == 2) | (seg_data == 3)).astype(np.float32)
        
        # Convert to tensors with channel dimension
        ct_tensor = torch.from_numpy(ct_norm)[None]           # [1,D,H,W]
        kidney_tensor = torch.from_numpy(combined_mask)[None]  # [1,D,H,W] combined mask
        tumor_tensor = torch.from_numpy(tumor_mask)[None]      # [1,D,H,W]
        cyst_tensor = torch.from_numpy(cyst_mask)[None]       # [1,D,H,W]
        
        # Save preprocessed tensors
        output_file = output_dir / f"{case_path.name}.pt"
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
        
        logger.info(f"Processed {case_path.name}:")
        logger.info(f"  Shape: {original_shape}")
        logger.info(f"  Spacing: {spacing}")
        logger.info(f"  Kidney voxels: {int(kidney_mask.sum())}")
        logger.info(f"  Tumor voxels: {int(tumor_mask.sum())}")
        logger.info(f"  Cyst voxels: {int(cyst_mask.sum())}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {case_path.name}: {str(e)}")
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
    logger.info(f"Found {len(case_folders)} cases in {data_dir}")
    
    # Process each case
    success_count = 0
    window_range = (args.window_min, args.window_max)
    
    for case_folder in tqdm(case_folders, desc="Processing cases"):
        if precompute_case(case_folder, output_dir, window_range):
            success_count += 1
            
    logger.info(f"\nPreprocessing complete!")
    logger.info(f"Successfully processed {success_count}/{len(case_folders)} cases")
    logger.info(f"Preprocessed volumes saved to {output_dir}")

if __name__ == "__main__":
    main()