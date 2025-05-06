import torch
from torch.utils.data import Dataset
from pathlib import Path
from preprocessing.preprocessor import KiTS23Preprocessor
from augmentations import KiTS23Augmenter
from typing import List, Tuple, Dict
import gc
import numpy as np # Needed for searchsorted
from tqdm import tqdm # For progress bar during init

class KiTS23Dataset(Dataset):
    def __init__(self, root_dir: str, config, preprocessor: KiTS23Preprocessor = None, train: bool = True, preprocess: bool = True):
        """
        Initializes the dataset.

        Args:
            root_dir (str): Path to the raw dataset directory (e.g., 'kits23/dataset').
            config: Configuration object.
            preprocessor (KiTS23Preprocessor, optional): Pre-initialized preprocessor. Defaults to None.
            train (bool, optional): Whether this is a training dataset (enables augmentations). Defaults to True.
            preprocess (bool, optional): Whether to run the preprocessing step. If False, assumes preprocessed files exist. Defaults to True.
        """
        self.root_dir = Path(root_dir)
        self.config = config
        self.preprocessor = preprocessor or KiTS23Preprocessor(config)
        self.augmenter = KiTS23Augmenter(config) if train else None
        
        # Define path for preprocessed patch files
        self.preprocessed_dir = Path(getattr(config, 'preprocessed_dir', 'preprocessed_patches'))
        self.preprocessed_dir.mkdir(exist_ok=True)

        self.patch_metadata: List[Tuple[Path, int]] = [] # Stores (path_to_pt_file, num_patches_in_file)
        self.cumulative_patches: List[int] = [0] # Cumulative count for indexing
        self._total_patches = 0

        if preprocess:
            print(f"Preprocessing dataset from {self.root_dir} and saving patches to {self.preprocessed_dir}...")
            all_case_paths = sorted([d for d in self.root_dir.iterdir() if d.is_dir() and d.name.startswith('case_')])
            
            for case_path in tqdm(all_case_paths, desc="Preprocessing Cases"):
                case_output_file = self.preprocessed_dir / f"{case_path.name}_patches.pt"
                
                # Skip if already processed (optional, but good for resuming)
                # Note: If you change preprocessing parameters, you might need to delete old files
                if case_output_file.exists():
                    try:
                        # Quick load to check number of patches if metadata needs rebuilding
                        # This adds overhead but ensures consistency if script was interrupted
                        # A separate metadata file could optimize this.
                        num_patches = len(torch.load(case_output_file))
                        if num_patches > 0:
                             print(f"Found existing {case_output_file} with {num_patches} patches.")
                             self.patch_metadata.append((case_output_file, num_patches))
                             self._total_patches += num_patches
                             self.cumulative_patches.append(self._total_patches)
                        else:
                             print(f"Found existing empty patch file {case_output_file}, skipping.")
                        continue # Skip reprocessing
                    except Exception as e:
                        print(f"Error loading existing file {case_output_file}, will re-process: {e}")
                        # Fall through to reprocessing

                # --- Actual Preprocessing ---
                print(f"Processing case {case_path.name}") # Keep some logging inside tqdm
                case_patches = self.preprocessor.preprocess_case(case_path)
                
                if case_patches:
                    num_patches = len(case_patches)
                    print(f"Saving {num_patches} patches for {case_path.name} to {case_output_file}...")
                    try:
                        torch.save(case_patches, case_output_file)
                        self.patch_metadata.append((case_output_file, num_patches))
                        self._total_patches += num_patches
                        self.cumulative_patches.append(self._total_patches)
                    except Exception as e:
                        print(f"ERROR saving patches for {case_path.name}: {e}")
                    
                    # Clear patches from memory for this case
                    del case_patches
                else:
                    print(f"No patches extracted or case skipped for {case_path.name}.")

                # Force garbage collection periodically
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            print(f"Preprocessing complete. Total patches saved: {self._total_patches}")

        else: # Not preprocessing, load existing metadata
            print(f"Loading existing preprocessed patches from {self.preprocessed_dir}...")
            if not self.preprocessed_dir.exists():
                 raise FileNotFoundError(f"Preprocessed directory not found: {self.preprocessed_dir}. Run with preprocess=True first.")
                 
            all_patch_files = sorted(self.preprocessed_dir.glob("case_*_patches.pt"))
            if not all_patch_files:
                 raise FileNotFoundError(f"No preprocessed patch files found in {self.preprocessed_dir}. Run with preprocess=True first.")

            for patch_file in tqdm(all_patch_files, desc="Loading Patch Metadata"):
                try:
                    # Load only the length, not the whole data yet
                    # This is still inefficient if files are huge, consider saving metadata separately
                    num_patches = len(torch.load(patch_file))
                    if num_patches > 0:
                        self.patch_metadata.append((patch_file, num_patches))
                        self._total_patches += num_patches
                        self.cumulative_patches.append(self._total_patches)
                except Exception as e:
                    print(f"Warning: Could not load or get length from {patch_file}, skipping: {e}")
            
            print(f"Found {len(self.patch_metadata)} patch files with a total of {self._total_patches} patches.")
            if self._total_patches == 0:
                 print("Warning: No valid patches found in the preprocessed directory.")

        # Convert cumulative_patches to numpy array for faster searching
        self.cumulative_patches = np.array(self.cumulative_patches, dtype=np.int64)


    def __len__(self) -> int:
        # The length of the dataset is the total number of individual patches
        return self._total_patches

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self._total_patches:
            raise IndexError(f"Index {idx} out of range for dataset with length {self._total_patches}")

        # Find which file this patch index belongs to
        # searchsorted returns the index 'file_idx' such that cumulative_patches[file_idx-1] <= idx < cumulative_patches[file_idx]
        file_idx = np.searchsorted(self.cumulative_patches, idx, side='right')
        
        # Calculate the index within that specific file
        start_idx_of_file = self.cumulative_patches[file_idx - 1]
        patch_idx_in_file = idx - start_idx_of_file
        
        # Get the path to the file
        file_path, num_patches_in_file = self.patch_metadata[file_idx - 1] # file_idx is 1-based because of the initial 0

        # Load the patches from the file
        try:
            patches_in_file = torch.load(file_path)
        except Exception as e:
             print(f"ERROR loading patch file {file_path} for index {idx}: {e}")
             # Return dummy data or raise error? Let's raise for now.
             raise RuntimeError(f"Failed to load patch file {file_path}") from e

        if patch_idx_in_file >= len(patches_in_file):
             # This shouldn't happen if cumulative counts are correct, but check for safety
             raise IndexError(f"Calculated patch index {patch_idx_in_file} out of range for file {file_path} with {len(patches_in_file)} patches (global index {idx})")

        # Get the specific patch
        image, mask = patches_in_file[patch_idx_in_file]
        
        # Clear the loaded list to save memory, only keep the selected patch
        del patches_in_file
        # gc.collect() # Might be too slow to collect garbage in every __getitem__

        # Apply augmentations if needed
        if self.augmenter is not None:
            image, mask = self.augmenter(image, mask)
            
        # Clone tensors to make them resizable
        image = image.clone()
        mask = mask.clone()
            
        return image, mask

    @staticmethod
    def collate_fn(batch):
        """Custom collate function to handle variable sized tensors."""
        images, masks = zip(*batch)
        
        # Stack images and masks separately
        images = torch.stack([img.contiguous() for img in images], dim=0)
        masks = torch.stack([msk.contiguous() for msk in masks], dim=0)
        
        return images, masks