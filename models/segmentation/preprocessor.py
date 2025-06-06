import os
import torch
import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import logging
from tqdm import tqdm
import SimpleITK as sitk
from scipy.ndimage import zoom
import argparse

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "/workspace/RCC_Project/preprocessed_volumes"

class VolumePreprocessor:
    """Preprocess KiTS23 imaging data and segmentations"""
    
    def __init__(self,
                target_spacing: Tuple[float, float, float] = (3.0, 1.5, 1.5),
                target_size: Tuple[int, int, int] = (128, 256, 256),
                ct_clip_range: Tuple[float, float] = (-1024, 1024),
                normalize: bool = True):
        """
        Args:
            target_spacing: Output voxel spacing (Z,Y,X) in mm
            target_size: Output volume dimensions (D,H,W)
            ct_clip_range: HU value range for CT clipping
            normalize: Whether to normalize CT values to [0,1]
        """
        self.target_spacing = target_spacing
        self.target_size = target_size
        self.ct_clip_range = ct_clip_range
        self.normalize = normalize

    def process_case(self, 
                   kits_case_dir: Path,
                   output_dir: Path,
                   case_id: str) -> bool:
        """
        Process single KiTS23 case
        
        Args:
            kits_case_dir: Path to KiTS23 case directory containing imaging.nii.gz and segmentation.nii.gz
            output_dir: Output directory for preprocessed .pt files
            case_id: Case identifier (e.g., "case_00123")
            
        Returns:
            bool: True if processing succeeded
        """
        try:
            # Load KiTS23 data
            image_path = kits_case_dir / "imaging.nii.gz"
            seg_path = kits_case_dir / "segmentation.nii.gz"
            
            if not image_path.exists() or not seg_path.exists():
                logger.error(f"Missing files for {case_id}")
                return False
                
            # Load CT volume and header information
            ct_nii = nib.load(str(image_path))
            ct_data = ct_nii.get_fdata().astype(np.float32)
            original_affine = ct_nii.affine
            
            # Load segmentation
            seg_nii = nib.load(str(seg_path))
            seg_data = seg_nii.get_fdata().astype(np.float32)
            
            # Store original shape for reference
            original_shape = ct_data.shape
            
            # Extract masks from segmentation:
            # KiTS23 labels: 1=kidney, 2=tumor, 3=cyst
            kidney_mask = (seg_data == 1).astype(np.float32)
            tumor_mask = (seg_data == 2).astype(np.float32)
            
            # Get original spacing from header
            orig_spacing = np.array(ct_nii.header.get_zooms())
            
            # Resample to target spacing
            ct_resampled = self._resample_volume(
                ct_data, 
                orig_spacing=orig_spacing,
                target_spacing=self.target_spacing
            )
            
            kidney_resampled = self._resample_volume(
                kidney_mask,
                orig_spacing=orig_spacing,
                target_spacing=self.target_spacing,
                is_mask=True
            )
            
            tumor_resampled = self._resample_volume(
                tumor_mask,
                orig_spacing=orig_spacing,
                target_spacing=self.target_spacing,
                is_mask=True
            )
            
            # Store intermediate shape after resampling
            resampled_shape = ct_resampled.shape
            
            # Crop/pad to target size
            ct_final = self._resize_volume(ct_resampled, self.target_size)
            kidney_final = self._resize_volume(kidney_resampled, self.target_size)
            tumor_final = self._resize_volume(tumor_resampled, self.target_size)
            
            # Clip & normalize CT
            ct_final = np.clip(ct_final, self.ct_clip_range[0], self.ct_clip_range[1])
            if self.normalize:
                ct_final = (ct_final - self.ct_clip_range[0]) / (self.ct_clip_range[1] - self.ct_clip_range[0])
            
            # Convert to tensors with channel dimension
            ct_tensor = torch.from_numpy(ct_final)[None]  # [1,D,H,W]
            kidney_tensor = torch.from_numpy(kidney_final)[None]  # [1,D,H,W]
            tumor_tensor = torch.from_numpy(tumor_final)[None]  # [1,D,H,W]
            
            # Save preprocessed data with full metadata for both patch and volume training
            output_path = output_dir / f"{case_id}.pt"
            torch.save({
                # Tensors for training
                'image': ct_tensor,          # For both patch and volume training
                'kidney': kidney_tensor,     # For Phase 2 kidney-aware training
                'tumor': tumor_tensor,       # For tumor segmentation
                
                # Processing parameters
                'spacing': self.target_spacing,
                'size': self.target_size,
                'clip_range': self.ct_clip_range,
                'normalized': self.normalize,
                
                # Original metadata for volume training/evaluation
                'original_spacing': orig_spacing.tolist(),
                'original_shape': original_shape,
                'resampled_shape': resampled_shape,
                'original_affine': original_affine.tolist(),
                
                # Case information
                'case_id': case_id,
                'source': 'KiTS23'
            }, output_path)
            
            # Log statistics
            logger.info(f"Processed {case_id}")
            logger.info(f"  CT shape: {ct_tensor.shape}")
            logger.info(f"  Original shape: {original_shape}")
            logger.info(f"  Original spacing: {orig_spacing.tolist()}")
            logger.info(f"  Target shape: {self.target_size}")
            logger.info(f"  Target spacing: {self.target_spacing}")
            logger.info(f"  Kidney voxels: {kidney_tensor.sum().item()}")
            logger.info(f"  Tumor voxels: {tumor_tensor.sum().item()}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing {case_id}: {str(e)}")
            return False

    def process_dataset(self,
                      kits_root: Path,
                      output_dir: Optional[Path] = None,
                      num_cases: Optional[int] = None):
        """
        Process all cases in KiTS23 dataset
        
        Args:
            kits_root: Path to KiTS23 dataset root containing case_XXXXX folders
            output_dir: Output directory for preprocessed files (default: /workspace/RCC_Project/preprocessed_volumes)
            num_cases: Optional limit on number of cases to process
        """
        # Use default output directory if none specified
        if output_dir is None:
            output_dir = Path(DEFAULT_OUTPUT_DIR)
            
        # Create output directory
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Find all case directories
        case_dirs = sorted(list(kits_root.glob("case_*")))
        if num_cases:
            case_dirs = case_dirs[:num_cases]
            
        logger.info(f"Found {len(case_dirs)} cases")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Target spacing: {self.target_spacing}")
        logger.info(f"Target size: {self.target_size}")
        
        # Process each case
        success = 0
        for case_dir in tqdm(case_dirs):
            if self.process_case(case_dir, output_dir, case_dir.name):
                success += 1
                
        logger.info(f"Successfully processed {success}/{len(case_dirs)} cases")
        logger.info(f"Preprocessed files saved to: {output_dir}")

    def _resample_volume(self,
                       volume: np.ndarray,
                       orig_spacing: np.ndarray,
                       target_spacing: Tuple[float, float, float],
                       is_mask: bool = False) -> np.ndarray:
        """Resample volume to target spacing"""
        # Calculate resize factors
        resize_factor = orig_spacing / target_spacing
        
        if is_mask:
            # Nearest neighbor interpolation for masks
            return zoom(volume, resize_factor, order=0)
        else:
            # Linear interpolation for images
            return zoom(volume, resize_factor, order=1)

    def _resize_volume(self,
                     volume: np.ndarray,
                     target_size: Tuple[int, int, int]) -> np.ndarray:
        """Center crop/pad volume to target size"""
        D, H, W = volume.shape
        TD, TH, TW = target_size
        
        # Calculate crop/pad amounts
        d_start = max(0, (D - TD) // 2)
        h_start = max(0, (H - TH) // 2)
        w_start = max(0, (W - TW) // 2)
        
        d_end = min(D, d_start + TD)
        h_end = min(H, h_start + TH)
        w_end = min(W, w_start + TW)
        
        # Crop
        cropped = volume[d_start:d_end, h_start:h_end, w_start:w_end]
        
        # Pad if necessary
        d_pad = (max(0, (TD - cropped.shape[0]) // 2),
                max(0, TD - cropped.shape[0] - max(0, (TD - cropped.shape[0]) // 2)))
        h_pad = (max(0, (TH - cropped.shape[1]) // 2),
                max(0, TH - cropped.shape[1] - max(0, (TH - cropped.shape[1]) // 2)))
        w_pad = (max(0, (TW - cropped.shape[2]) // 2),
                max(0, TW - cropped.shape[2] - max(0, (TW - cropped.shape[2]) // 2)))
        
        padded = np.pad(cropped, (d_pad, h_pad, w_pad), mode='constant')
        
        return padded

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kits_dir', type=str, required=True,
                      help='Path to KiTS23 dataset root')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                      help='Output directory for preprocessed files')
    parser.add_argument('--target_spacing', type=float, nargs=3,
                      default=[3.0, 1.5, 1.5],
                      help='Target voxel spacing (Z,Y,X) in mm')
    parser.add_argument('--target_size', type=int, nargs=3,
                      default=[128, 256, 256],
                      help='Target volume dimensions (D,H,W)')
    parser.add_argument('--num_cases', type=int,
                      help='Number of cases to process (default: all)')
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(args.output_dir) / 'preprocessing.log')
        ]
    )
    
    # Create preprocessor
    preprocessor = VolumePreprocessor(
        target_spacing=tuple(args.target_spacing),
        target_size=tuple(args.target_size)
    )
    
    # Process dataset
    preprocessor.process_dataset(
        Path(args.kits_dir),
        Path(args.output_dir),
        args.num_cases
    )

if __name__ == '__main__':
    main()