import os
from pathlib import Path
import nibabel as nib
import numpy as np
from totalsegmentator.python_api import totalsegmentator
from tempfile import TemporaryDirectory
import shutil
import subprocess

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
        
        if self.debug:
            # Print environment information
            print("\nEnvironment Information:")
            print(f"Current working directory: {os.getcwd()}")
            print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
            print(f"USER: {os.environ.get('USER', 'Not set')}")
            print(f"HOME: {os.environ.get('HOME', 'Not set')}")
            
            # Check TotalSegmentator installation
            try:
                import totalsegmentator
                print(f"TotalSegmentator version: {totalsegmentator.__version__}")
                print(f"TotalSegmentator path: {totalsegmentator.__file__}")
            except Exception as e:
                print(f"Error getting TotalSegmentator info: {str(e)}")
        
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
                print(f"\nRunning TotalSegmentator:")
                print(f"Input path: {input_path}")
                print(f"Output directory: {output_path}")
                print(f"Exists - Input: {Path(input_path).exists()}")
                print(f"Exists - Output dir: {Path(output_path).exists()}")
                
                # Print directory contents before
                print("\nOutput directory contents before segmentation:")
                self._print_dir_contents(output_path)
                
            # Run TotalSegmentator with kidney-specific settings
            totalsegmentator(
                input=str(input_path),
                output=str(output_path),
                fast=self.model_type == "fast",
                roi_subset=["kidney_right", "kidney_left"],
                ml=True,  # Use machine learning model
                verbose=self.debug  # Enable verbose output in debug mode
            )
            
            if self.debug:
                print("\nSegmentation completed")
                print("\nOutput directory contents after segmentation:")
                self._print_dir_contents(output_path)
                
                # Try running ls directly
                try:
                    result = subprocess.run(['ls', '-la', str(output_path)], 
                                         capture_output=True, text=True)
                    print("\nDirectory listing using ls -la:")
                    print(result.stdout)
                except Exception as e:
                    print(f"Error running ls: {str(e)}")
                
            return True
            
        except Exception as e:
            print(f"Error during segmentation: {str(e)}")
            if self.debug:
                import traceback
                print("Full traceback:")
                print(traceback.format_exc())
            return False
    
    def _print_dir_contents(self, directory):
        """Recursively print directory contents"""
        directory = Path(directory)
        if not directory.exists():
            print(f"Directory does not exist: {directory}")
            return
            
        try:
            print(f"\nContents of {directory}:")
            for item in directory.rglob('*'):
                try:
                    print(f"  {'D' if item.is_dir() else 'F'} {item.relative_to(directory)}")
                except Exception as e:
                    print(f"Error accessing {item}: {str(e)}")
        except Exception as e:
            print(f"Error listing directory: {str(e)}")
    
    def _combine_kidney_masks(self, seg_dir):
        """Combine individual kidney segmentations into a single mask"""
        combined_mask = None
        seg_dir = Path(seg_dir)
        
        if self.debug:
            print("\nAttempting to combine kidney masks:")
            print(f"Segmentation directory: {seg_dir}")
            print("Available files:")
            self._print_dir_contents(seg_dir)
        
        # Look for all possible naming patterns
        possible_patterns = [
            # Original pattern
            {'kidney_right': 'kidney_right.nii.gz', 
             'kidney_left': 'kidney_left.nii.gz'},
            # Alternative patterns
            {'kidney_right': 'total_segmentator_kidney_right.nii.gz', 
             'kidney_left': 'total_segmentator_kidney_left.nii.gz'},
            {'kidney_right': '1.nii.gz',  # Sometimes numbered
             'kidney_left': '2.nii.gz'},
            {'kidney_right': 'kidney_right_seg.nii.gz',
             'kidney_left': 'kidney_left_seg.nii.gz'}
        ]
        
        found_files = False
        for pattern in possible_patterns:
            if self.debug:
                print(f"\nTrying pattern: {pattern}")
                for kidney, filename in pattern.items():
                    path = seg_dir / filename
                    print(f"Checking {path}: {path.exists()}")
                    
            if all((seg_dir / filename).exists() for filename in pattern.values()):
                if self.debug:
                    print(f"Found kidney files using pattern: {pattern}")
                filenames = pattern
                found_files = True
                break
                
        if not found_files:
            if self.debug:
                print("Could not find kidney segmentation files with any known pattern")
            return None
            
        # Process found files
        for kidney, filename in filenames.items():
            mask_path = seg_dir / filename
            if not mask_path.exists():
                if self.debug:
                    print(f"Warning: {mask_path} not found")
                continue
                
            try:
                if self.debug:
                    print(f"\nProcessing {kidney} mask:")
                    print(f"Loading from: {mask_path}")
                    
                mask = nib.load(mask_path)
                mask_data = mask.get_fdata()
                
                if self.debug:
                    print(f"Loaded mask shape: {mask_data.shape}")
                    print(f"Value range: [{mask_data.min():.2f}, {mask_data.max():.2f}]")
                    print(f"Unique values: {np.unique(mask_data)}")
                
                # Initialize combined mask if needed
                if combined_mask is None:
                    combined_mask = np.zeros_like(mask_data)
                
                # Add this kidney's mask
                combined_mask[mask_data > 0] = 1
                
            except Exception as e:
                print(f"Error processing {kidney} mask: {str(e)}")
                if self.debug:
                    import traceback
                    print(traceback.format_exc())
                continue
        
        if combined_mask is not None and self.debug:
            print(f"\nCombined mask statistics:")
            print(f"Shape: {combined_mask.shape}")
            print(f"Value range: [{combined_mask.min():.2f}, {combined_mask.max():.2f}]")
            print(f"Unique values: {np.unique(combined_mask)}")
            print(f"Non-zero voxels: {np.count_nonzero(combined_mask)}")
        
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
            
            if self.debug:
                print(f"\nTemporary directory created: {temp_dir}")
                print(f"Directory exists: {temp_dir.exists()}")
            
            # Run segmentation
            if not self._run_segmentation(img_path, temp_dir):
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