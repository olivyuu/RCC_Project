import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Tuple, List
import torch
from tqdm import tqdm
import gc
from nibabel.processing import resample_from_to

class KiTS23Preprocessor:
    def __init__(self, config):
        print("\nDebug: Initializing KiTS23Preprocessor")
        print("Debug: Initial setup")
        methods = [m for m in dir(self.__class__) if not m.startswith('__')]
        print(f"Debug: Available class methods: {methods}")
        self.config = config
        self.patch_size = config.patch_size
        self.stride = config.patch_stride
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processing_dtype = np.float32
        self.processing_dtype_itemsize = np.dtype(self.processing_dtype).itemsize
        self.volume_max_dim = getattr(config, 'vol_max_dim', (128, 256, 256))
        self.max_estimated_memory_mb = getattr(config, 'max_estimated_memory_mb', config.max_image_size_mb * 1.5)
        print(f"Preprocessor initialized with patch size: {self.patch_size}")
        print(f"Target processing dtype: {self.processing_dtype}")
        print(f"Max estimated memory limit (for {self.processing_dtype}): {self.max_estimated_memory_mb:.1f} MB")
        print("Debug: Available methods:", [method for method in dir(self) if not method.startswith('__')])

    def preprocess_volume(self, case_path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Preprocess a case as a full volume without patch extraction.
        
        Args:
            case_path: Path to the case directory
            
        Returns:
            Tuple of (image_tensor, kidney_mask_tensor, tumor_mask_tensor) or (None, None, None) if processing fails
        """
        print(f"Debug: Starting volume preprocessing for {case_path}")
        image_chunks, kidney_mask_chunks, tumor_mask_chunks = self.load_case(case_path)
        
        if not image_chunks:
            print(f"Failed to load case {case_path.name}")
            return None, None, None
            
        try:
            print(f"Debug: Concatenating {len(image_chunks)} chunks")
            # Concatenate chunks along depth dimension
            img_volume = np.concatenate(image_chunks, axis=0)
            kidney_mask_volume = np.concatenate(kidney_mask_chunks, axis=0)
            tumor_mask_volume = np.concatenate(tumor_mask_chunks, axis=0)
            
            print(f"Debug: Converting to tensors, volume shape: {img_volume.shape}")
            # Convert to tensors
            img_tensor = torch.from_numpy(img_volume).float().unsqueeze(0)  # Add channel dimension
            kidney_mask_tensor = torch.from_numpy(kidney_mask_volume).float().unsqueeze(0)
            tumor_mask_tensor = torch.from_numpy(tumor_mask_volume).float().unsqueeze(0)
            
            # Resize if needed
            if any(s > m for s, m in zip(img_tensor.shape[-3:], self.volume_max_dim)):
                print(f"Resizing volume from {img_tensor.shape[-3:]} to {self.volume_max_dim}")
                img_tensor = torch.nn.functional.interpolate(
                    img_tensor.unsqueeze(0),  # Add batch dimension
                    size=self.volume_max_dim,
                    mode='trilinear',
                    align_corners=False
                ).squeeze(0)  # Remove batch dimension
                
                kidney_mask_tensor = torch.nn.functional.interpolate(
                    kidney_mask_tensor.unsqueeze(0),
                    size=self.volume_max_dim,
                    mode='nearest'
                ).squeeze(0)
                
                tumor_mask_tensor = torch.nn.functional.interpolate(
                    tumor_mask_tensor.unsqueeze(0),
                    size=self.volume_max_dim,
                    mode='nearest'
                ).squeeze(0)
            
            # Clear intermediate data
            del image_chunks, kidney_mask_chunks, tumor_mask_chunks
            gc.collect()
            
            print(f"Debug: Successfully preprocessed volume, final shape: {img_tensor.shape}")
            return img_tensor, kidney_mask_tensor, tumor_mask_tensor
            
        except Exception as e:
            print(f"Error preprocessing volume {case_path.name}: {e}")
            return None, None, None

    def load_case(self, case_path: Path) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Load and preprocess a case."""
        print(f"\nLoading case: {case_path.name}")
        img_path = case_path / "imaging.nii.gz"
        mask_path = case_path / "segmentation.nii.gz"
            
        if not img_path.exists() or not mask_path.exists():
            print(f"Could not find imaging or segmentation files for {case_path.name}")
            return [], [], []

        img_obj = None
        mask_obj = None
        try:
            # Load data
            img_obj = nib.load(str(img_path))
            mask_obj = nib.load(str(mask_path))
            shape = img_obj.shape

            # Process in chunks
            chunk_size = min(shape[0], max(32, self.patch_size[0]))
            num_chunks = (shape[0] + chunk_size - 1) // chunk_size

            image_chunks = []
            kidney_mask_chunks = []
            tumor_mask_chunks = []

            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, shape[0])
                
                # Load and normalize chunk
                img_chunk = np.asarray(img_obj.dataobj[start_idx:end_idx], dtype=self.processing_dtype)
                mask_chunk = np.asarray(mask_obj.dataobj[start_idx:end_idx], dtype=self.processing_dtype)

                # Normalize image chunk
                chunk_min = np.min(img_chunk)
                chunk_max = np.max(img_chunk)
                if chunk_max > chunk_min:
                    img_chunk = (img_chunk - chunk_min) / (chunk_max - chunk_min)
                else:
                    img_chunk = np.zeros_like(img_chunk)

                # Separate kidney (value 1) and tumor (value 2) masks
                kidney_mask = (mask_chunk == 1).astype(self.processing_dtype)
                tumor_mask = (mask_chunk == 2).astype(self.processing_dtype)

                image_chunks.append(img_chunk)
                kidney_mask_chunks.append(kidney_mask)
                tumor_mask_chunks.append(tumor_mask)

            print(f"Successfully processed {len(image_chunks)} chunks")
            return image_chunks, kidney_mask_chunks, tumor_mask_chunks

        except Exception as e:
            print(f"Failed during load_case for {case_path.name}: {e}")
            return [], [], []
        finally:
            del img_obj
            del mask_obj
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def preprocess_case(self, case_path: Path) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Preprocess a case and extract patches."""
        image_chunks, kidney_mask_chunks, tumor_mask_chunks = self.load_case(case_path)

        if not image_chunks:
            print(f"Warning: No image chunks loaded for {case_path.name}")
            return []

        # Extract patches
        patches = self.extract_patches(image_chunks, kidney_mask_chunks, tumor_mask_chunks)
        
        print(f"Extracted {len(patches)} patches from {case_path.name}")
        return patches

    def extract_patches(self, image_chunks: List[np.ndarray], kidney_mask_chunks: List[np.ndarray], 
                       tumor_mask_chunks: List[np.ndarray]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Extract patches from chunks."""
        if not image_chunks or not kidney_mask_chunks or not tumor_mask_chunks:
            return []

        patches = []
        h_range = image_chunks[0].shape[1] - self.patch_size[1] + 1
        w_range = image_chunks[0].shape[2] - self.patch_size[2] + 1

        for img_chunk, kidney_mask_chunk, tumor_mask_chunk in zip(image_chunks, kidney_mask_chunks, tumor_mask_chunks):
            for d in range(0, img_chunk.shape[0] - self.patch_size[0] + 1, self.stride[0]):
                for h in range(0, h_range, self.stride[1]):
                    for w in range(0, w_range, self.stride[2]):
                        img_patch = img_chunk[d:d+self.patch_size[0],
                                          h:h+self.patch_size[1],
                                          w:w+self.patch_size[2]]
                        kidney_mask_patch = kidney_mask_chunk[d:d+self.patch_size[0],
                                                         h:h+self.patch_size[1],
                                                         w:w+self.patch_size[2]]
                        tumor_mask_patch = tumor_mask_chunk[d:d+self.patch_size[0],
                                                       h:h+self.patch_size[1],
                                                       w:w+self.patch_size[2]]

                        if tumor_mask_patch.sum() > 0:  # Only keep patches with tumor annotations
                            # Stack image and kidney mask as input channels
                            input_patch = np.stack([img_patch, kidney_mask_patch], axis=0)
                            input_tensor = torch.from_numpy(input_patch).float()
                            tumor_mask_tensor = torch.from_numpy(tumor_mask_patch).float().unsqueeze(0)
                            patches.append((input_tensor, tumor_mask_tensor))

        return patches