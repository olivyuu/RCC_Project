import os
from pathlib import Path
import nibabel as nib
import numpy as np
from totalsegmentator.python_api import totalsegmentator
from tempfile import TemporaryDirectory
import shutil
import glob

class KidneySegmentor:
    """Helper class to handle kidney segmentation using TotalSegmentator"""
    
    def __init__(self, model_type="fast", output_dir="segmentations", debug=False):
        self.model_type = model_type
        self.output_dir = Path(output_dir)
        self.debug = debug
        
        if self.debug:
            print("\nEnvironment Information:")
            print(f"Current working directory: {os.getcwd()}")
            print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
            print(f"USER: {os.environ.get('USER', 'Not set')}")
            print(f"HOME: {os.environ.get('HOME', 'Not set')}")
            print(f"Temporary directory: {os.environ.get('TMPDIR', '/tmp')}")
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _run_segmentation(self, input_path, output_path):
        """Run TotalSegmentator on the input image"""
        try:
            if self.debug:
                print(f"\nRunning TotalSegmentator:")
                print(f"Input path: {input_path}")
                print(f"Output directory: {output_path}")
                
            # Create output directory
            Path(output_path).mkdir(parents=True, exist_ok=True)
                
            # Run TotalSegmentator
            totalsegmentator(
                input=str(input_path),
                output=str(output_path / "segmentations.nii.gz"),  # Single output file
                fast=self.model_type == "fast",
                roi_subset=["kidney_right", "kidney_left"],
                ml=True,
                verbose=self.debug
            )
            
            if self.debug:
                print("\nSegmentation completed")
                print("\nLooking for output files:")
                # Search all temporary directories
                tmp_dirs = glob.glob("/tmp/nnunet_tmp_*")
                for tmp_dir in tmp_dirs:
                    print(f"\nChecking temporary directory: {tmp_dir}")
                    print("Contents:")
                    os.system(f"ls -la {tmp_dir}")
                
                print("\nOutput directory contents:")
                os.system(f"ls -la {output_path}")
                
            return True
            
        except Exception as e:
            print(f"Error during segmentation: {str(e)}")
            if self.debug:
                import traceback
                print("Full traceback:")
                print(traceback.format_exc())
            return False
    
    def _extract_kidney_masks(self, seg_file):
        """Extract kidney masks from the segmentation file"""
        try:
            if self.debug:
                print(f"\nExtracting kidney masks from: {seg_file}")
                
            # Load segmentation file
            seg = nib.load(seg_file)
            seg_data = seg.get_fdata()
            
            if self.debug:
                print(f"Segmentation shape: {seg_data.shape}")
                print(f"Unique labels: {np.unique(seg_data)}")
            
            # Create binary mask for kidneys
            # Label values from TotalSegmentator documentation:
            # Kidney (right): 2
            # Kidney (left): 3
            combined_mask = np.zeros_like(seg_data)
            combined_mask[(seg_data == 2) | (seg_data == 3)] = 1
            
            if self.debug:
                print(f"Combined mask shape: {combined_mask.shape}")
                print(f"Non-zero voxels: {np.count_nonzero(combined_mask)}")
            
            return combined_mask
            
        except Exception as e:
            print(f"Error extracting kidney masks: {str(e)}")
            if self.debug:
                import traceback
                print(traceback.format_exc())
            return None
    
    def get_kidney_mask(self, img_path, case_id=None):
        """Generate kidney segmentation mask for the given image"""
        img_path = Path(img_path)
        if not img_path.exists():
            print(f"Input image not found: {img_path}")
            return None
            
        # Use image name as case_id if not provided
        if case_id is None:
            case_id = img_path.stem
            
        # Check if we already have the segmentation
        output_path = self.output_dir / case_id / "kidney_mask.nii.gz"
        if output_path.exists():
            if self.debug:
                print(f"Loading existing segmentation from {output_path}")
            try:
                mask = nib.load(output_path)
                return mask.get_fdata()
            except:
                if self.debug:
                    print("Failed to load existing segmentation, regenerating...")
                    
        # Create temporary directory for TotalSegmentator output
        with TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            
            if self.debug:
                print(f"\nTemporary directory created: {temp_dir}")
            
            # Run segmentation
            if not self._run_segmentation(img_path, temp_dir):
                return None
            
            # Look for segmentation file
            seg_file = temp_dir / "segmentations.nii.gz"
            if not seg_file.exists():
                print(f"Segmentation file not found: {seg_file}")
                return None
            
            # Extract kidney masks
            combined_mask = self._extract_kidney_masks(seg_file)
            if combined_mask is None:
                print("Failed to extract kidney masks")
                return None
            
            # Save the combined mask
            output_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(
                nib.Nifti1Image(combined_mask, np.eye(4)),
                output_path
            )
            
            if self.debug:
                print(f"Saved combined kidney mask to {output_path}")
            
            return combined_mask
            
    def clear_cache(self):
        """Clear the segmentation cache"""
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
            self.output_dir.mkdir(parents=True)
            if self.debug:
                print(f"Cleared segmentation cache in {self.output_dir}")