import torch
from torch.utils.data import Dataset
from pathlib import Path
import gc
import numpy as np
from tqdm import tqdm

from .preprocessor import SegmentationPreprocessor

class SegmentationDataset(Dataset):
    def __init__(self, root_dir: str, config, train: bool = True, preprocess: bool = True):
        """
        Dataset for tumor segmentation with TotalSegmentor kidney localization support.

        Args:
            root_dir (str): Path to the raw dataset directory (e.g., 'kits23/dataset')
            config: Configuration object
            train (bool): Whether this is a training dataset (enables augmentations)
            preprocess (bool): Whether to run preprocessing. If False, assumes preprocessed files exist.
        """
        self.root_dir = Path(root_dir)
        self.config = config
        self.preprocessor = SegmentationPreprocessor(config)
        self.train = train
        
        # Define path for preprocessed patch files
        self.preprocessed_dir = Path(getattr(config, 'preprocessed_dir', 'preprocessed_patches'))
        self.preprocessed_dir.mkdir(exist_ok=True)

        self.patch_metadata = []  # Stores (path_to_pt_file, num_patches)
        self.cumulative_patches = [0]  # For indexing
        self._total_patches = 0

        if preprocess:
            print(f"Preprocessing dataset from {self.root_dir} and saving to {self.preprocessed_dir}...")
            all_case_paths = sorted([d for d in self.root_dir.iterdir() 
                                   if d.is_dir() and d.name.startswith('case_')])
            
            for case_path in tqdm(all_case_paths, desc="Preprocessing Cases"):
                case_output_file = self.preprocessed_dir / f"{case_path.name}_patches.pt"
                
                # Skip if already processed
                if case_output_file.exists():
                    try:
                        patches = torch.load(case_output_file)
                        num_patches = len(patches)
                        if num_patches > 0:
                            print(f"Found existing {case_output_file} with {num_patches} patches")
                            self.patch_metadata.append((case_output_file, num_patches))
                            self._total_patches += num_patches
                            self.cumulative_patches.append(self._total_patches)
                        continue
                    except Exception as e:
                        print(f"Error loading existing file {case_output_file}, will re-process: {e}")

                # Process new case
                print(f"Processing case {case_path.name}")
                case_patches = self.preprocessor.preprocess_case(case_path)
                
                if case_patches:
                    num_patches = len(case_patches)
                    print(f"Saving {num_patches} patches for {case_path.name}")
                    try:
                        torch.save(case_patches, case_output_file)
                        self.patch_metadata.append((case_output_file, num_patches))
                        self._total_patches += num_patches
                        self.cumulative_patches.append(self._total_patches)
                    except Exception as e:
                        print(f"ERROR saving patches for {case_path.name}: {e}")
                    
                    # Clear patches from memory
                    del case_patches
                else:
                    print(f"No patches extracted for {case_path.name}")

                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            print(f"Preprocessing complete. Total patches: {self._total_patches}")

        else:  # Load existing preprocessed data
            print(f"Loading existing preprocessed patches from {self.preprocessed_dir}...")
            if not self.preprocessed_dir.exists():
                raise FileNotFoundError(
                    f"Preprocessed directory not found: {self.preprocessed_dir}. "
                    "Run with preprocess=True first."
                )
                
            all_patch_files = sorted(self.preprocessed_dir.glob("case_*_patches.pt"))
            if not all_patch_files:
                raise FileNotFoundError(
                    f"No preprocessed patch files found in {self.preprocessed_dir}. "
                    "Run with preprocess=True first."
                )

            for patch_file in tqdm(all_patch_files, desc="Loading Patch Metadata"):
                try:
                    num_patches = len(torch.load(patch_file))
                    if num_patches > 0:
                        self.patch_metadata.append((patch_file, num_patches))
                        self._total_patches += num_patches
                        self.cumulative_patches.append(self._total_patches)
                except Exception as e:
                    print(f"Warning: Could not load {patch_file}, skipping: {e}")
            
            print(f"Found {len(self.patch_metadata)} files with {self._total_patches} total patches")

        # Convert cumulative_patches to numpy array for faster searching
        self.cumulative_patches = np.array(self.cumulative_patches, dtype=np.int64)

    def __len__(self) -> int:
        return self._total_patches

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self._total_patches:
            raise IndexError(f"Index {idx} out of range for dataset with {self._total_patches} patches")

        # Find which file contains this patch
        file_idx = np.searchsorted(self.cumulative_patches, idx, side='right')
        
        # Calculate the index within that file
        start_idx_of_file = self.cumulative_patches[file_idx - 1]
        patch_idx_in_file = idx - start_idx_of_file
        
        # Get the path to the file
        file_path, num_patches = self.patch_metadata[file_idx - 1]

        # Load the patches
        try:
            patches = torch.load(file_path)
        except Exception as e:
            print(f"ERROR loading patch file {file_path} for index {idx}: {e}")
            raise RuntimeError(f"Failed to load patch file {file_path}") from e

        if patch_idx_in_file >= len(patches):
            raise IndexError(
                f"Calculated patch index {patch_idx_in_file} out of range "
                f"for file {file_path} with {len(patches)} patches (global index {idx})"
            )

        # Get the specific patch
        input_patch, target_patch = patches[patch_idx_in_file]
        
        # Clear the loaded list to save memory
        del patches

        return input_patch.clone(), target_patch.clone()

    @staticmethod
    def collate_fn(batch):
        """Custom collate function to handle patches"""
        inputs, targets = zip(*batch)
        
        # Stack inputs and targets
        inputs = torch.stack([inp for inp in inputs], dim=0)
        targets = torch.stack([tgt for tgt in targets], dim=0)
        
        return inputs, targets