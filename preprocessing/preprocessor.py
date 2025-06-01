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
        self.config = config
        self.patch_size = config.patch_size
        self.stride = config.patch_stride
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processing_dtype = np.float32
        self.processing_dtype_itemsize = np.dtype(self.processing_dtype).itemsize
        self.volume_max_dim = getattr(config, 'vol_max_dim', (128, 256, 256))
        self.max_estimated_memory_mb = getattr(config, 'max_estimated_memory_mb', config.max_image_size_mb * 1.5)

    def preprocess_volume(self, case_path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preprocess a case as a full volume with memory-efficient processing."""
        try:
            img_path = case_path / "imaging.nii.gz"
            mask_path = case_path / "segmentation.nii.gz"
            
            if not img_path.exists() or not mask_path.exists():
                print(f"Could not find imaging or segmentation files for {case_path.name}")
                return None, None, None

            # Load headers only first
            img_obj = nib.load(str(img_path))
            mask_obj = nib.load(str(mask_path))
            shape = img_obj.shape

            # Calculate memory-efficient chunk size
            available_memory = self.max_estimated_memory_mb * 1024 * 1024  # Convert to bytes
            voxel_memory = self.processing_dtype_itemsize * 3  # For image and two masks
            max_voxels_per_chunk = available_memory // voxel_memory
            chunk_size = min(
                shape[0],
                max(16, self.patch_size[0]),  # Minimum chunk size
                max_voxels_per_chunk // (shape[1] * shape[2])  # Memory-based chunk size
            )
            num_chunks = (shape[0] + chunk_size - 1) // chunk_size

            # Initialize tensors for final output at target size
            d_scale = self.volume_max_dim[0] / shape[0]
            h_scale = self.volume_max_dim[1] / shape[1]
            w_scale = self.volume_max_dim[2] / shape[2]
            
            chunk_target_depth = max(1, int(chunk_size * d_scale))
            final_img = torch.zeros((1, *self.volume_max_dim), dtype=torch.float32)
            final_kidney = torch.zeros((1, *self.volume_max_dim), dtype=torch.float32)
            final_tumor = torch.zeros((1, *self.volume_max_dim), dtype=torch.float32)

            # Process chunks and resize immediately
            for i in tqdm(range(num_chunks), desc="Processing chunks"):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, shape[0])
                target_start = min(int(start_idx * d_scale), self.volume_max_dim[0])
                target_end = min(int(end_idx * d_scale), self.volume_max_dim[0])
                
                # Load and process chunk
                with torch.no_grad():  # Ensure no gradient computation
                    # Load chunk data
                    img_chunk = torch.from_numpy(
                        np.asarray(img_obj.dataobj[start_idx:end_idx], dtype=self.processing_dtype)
                    ).float()
                    mask_chunk = torch.from_numpy(
                        np.asarray(mask_obj.dataobj[start_idx:end_idx], dtype=self.processing_dtype)
                    )

                    # Extract masks
                    kidney_chunk = (mask_chunk == 1).float()
                    tumor_chunk = (mask_chunk == 2).float()
                    del mask_chunk
                    
                    # Add required dimensions for interpolation
                    img_chunk = img_chunk.unsqueeze(0).unsqueeze(0)
                    kidney_chunk = kidney_chunk.unsqueeze(0).unsqueeze(0)
                    tumor_chunk = tumor_chunk.unsqueeze(0).unsqueeze(0)
                    
                    # Resize chunk
                    img_chunk_resized = torch.nn.functional.interpolate(
                        img_chunk,
                        size=(target_end - target_start, self.volume_max_dim[1], self.volume_max_dim[2]),
                        mode='trilinear',
                        align_corners=False
                    )
                    
                    kidney_chunk_resized = torch.nn.functional.interpolate(
                        kidney_chunk,
                        size=(target_end - target_start, self.volume_max_dim[1], self.volume_max_dim[2]),
                        mode='nearest'
                    )
                    
                    tumor_chunk_resized = torch.nn.functional.interpolate(
                        tumor_chunk,
                        size=(target_end - target_start, self.volume_max_dim[1], self.volume_max_dim[2]),
                        mode='nearest'
                    )
                    
                    # Store in final tensors
                    final_img[0, target_start:target_end] = img_chunk_resized.squeeze(0).squeeze(0)
                    final_kidney[0, target_start:target_end] = kidney_chunk_resized.squeeze(0).squeeze(0)
                    final_tumor[0, target_start:target_end] = tumor_chunk_resized.squeeze(0).squeeze(0)
                    
                    # Clear memory
                    del img_chunk, kidney_chunk, tumor_chunk
                    del img_chunk_resized, kidney_chunk_resized, tumor_chunk_resized
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Global normalization of the image
            with torch.no_grad():
                img_min = final_img.min()
                img_max = final_img.max()
                if img_max > img_min:
                    final_img = (final_img - img_min) / (img_max - img_min)
                else:
                    final_img.zero_()

            return final_img, final_kidney, final_tumor

        except Exception as e:
            print(f"Error preprocessing volume {case_path.name}: {e}")
            return None, None, None
        finally:
            # Clean up original objects
            del img_obj
            del mask_obj
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def extract_patches(self, image: torch.Tensor, kidney_mask: torch.Tensor, 
                       tumor_mask: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Extract patches from preprocessed tensors."""
        if image is None or kidney_mask is None or tumor_mask is None:
            return []

        patches = []
        with torch.no_grad():
            h_range = image.shape[2] - self.patch_size[1] + 1
            w_range = image.shape[3] - self.patch_size[2] + 1

            for d in range(0, image.shape[1] - self.patch_size[0] + 1, self.stride[0]):
                for h in range(0, h_range, self.stride[1]):
                    for w in range(0, w_range, self.stride[2]):
                        img_patch = image[0, d:d+self.patch_size[0],
                                        h:h+self.patch_size[1],
                                        w:w+self.patch_size[2]]
                        kidney_patch = kidney_mask[0, d:d+self.patch_size[0],
                                                 h:h+self.patch_size[1],
                                                 w:w+self.patch_size[2]]
                        tumor_patch = tumor_mask[0, d:d+self.patch_size[0],
                                               h:h+self.patch_size[1],
                                               w:w+self.patch_size[2]]

                        if tumor_patch.sum() > 0:  # Only keep patches with tumor annotations
                            input_patch = torch.stack([img_patch, kidney_patch], dim=0)
                            patches.append((input_patch, tumor_patch.unsqueeze(0)))

                        # Periodically clear memory
                        if len(patches) % 100 == 0:
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

        return patches