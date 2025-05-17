import os
from pathlib import Path
import nibabel as nib
import numpy as np
from totalsegmentator.python_api import totalsegmentator  # Fixed import
from tempfile import TemporaryDirectory
import shutil

class KidneySegmentor:
    """Helper class to handle kidney segmentation using TotalSegmentator"""
    
    def __init__(self, model_type="fast", output_dir="segmentations", debug=False):
        """
        Initialize the kidney segmentor
        
        Args:
            model_type (str): Which TotalSegmentator model to use ('fast' or 'original')
            output_dir (str): Directory to save segmentation results
            debug (bool): Whether to print debug information
        """
        self.model_type = model_type
        self.output_dir = Path(output_dir)
        self.debug = debug
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Map of kidney labels in TotalSegmentator output
        self.KIDNEY_LABELS = {
            'kidney_right': 1,
            'kidney_left': 2
        }
    
    def _run_segmentation(self, input_path, output_path):
        """Run TotalSegmentator on the input image"""
        try:
            if self.debug:
                print(f"Running TotalSegmentator on {input_path}")
                print(f"Output directory: {output_path}")
                
            # Run TotalSegmentator with kidney-specific settings
            totalsegmentator(
                input=input_path,  # Changed from input_path to input
                output=output_path,  # Changed from output_path to output
                fast=self.model_type == "fast",
                roi_subset=["kidney_right", "kidney_left"],
                nr_thr=-1  # Use all available threads
            )
            
            if self.debug:
                print("Segmentation completed successfully")
                
            return True
            
        except Exception as e:
            print(f"Error during segmentation: {str(e)}")
            return False
    
    def _combine_kidney_masks(self, seg_dir):
        """Combine individual kidney segmentations into a single mask"""
        combined_mask = None
        
        for kidney, label in self.KIDNEY_LABELS.items():
            # Load individual kidney segmentation
            mask_path = seg_dir / f"{kidney}.nii.gz"
            if not mask_path.exists():
                if self.debug:
                    print(f"Warning: {mask_path} not found")
                continue
                
            try:
                mask = nib.load(mask_path)
                mask_data = mask.get_fdata()
                
                # Initialize combined mask if needed
                if combined_mask is None:
                    combined_mask = np.zeros_like(mask_data)
                
                # Add this kidney's mask
                combined_mask[mask_data > 0] = 1
                
            except Exception as e:
                print(f"Error processing {kidney} mask: {str(e)}")
                continue
        
        return combined_mask
    
    def get_kidney_mask(self, img_path, case_id=None):
        """
        Generate kidney segmentation mask for the given image
        
        Args:
            img_path (str or Path): Path to input CT image
            case_id (str, optional): Case identifier for saving results
            
        Returns:
            numpy.ndarray: Binary mask of kidney regions
            None if segmentation fails
        """
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
            
            # Run segmentation
            if not self._run_segmentation(str(img_path), str(temp_dir)):
                return None
            
            # Combine individual kidney masks
            combined_mask = self._combine_kidney_masks(temp_dir)
            if combined_mask is None:
                print("Failed to generate combined kidney mask")
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