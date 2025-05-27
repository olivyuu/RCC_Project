import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import torch
from tqdm import tqdm
import gc
from kidney_segmentor import KidneySegmentor

class SegmentationPreprocessor:
    def __init__(self, config):
        self.config = config
        self.patch_size = config.patch_size
        self.stride = config.patch_stride
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processing_dtype = np.float32
        
        # Initialize TotalSegmentor
        self.kidney_segmentor = KidneySegmentor(
            model_type="fast",
            output_dir=Path(config.data_dir) / "totalsegmentor_cache"
        )

    def preprocess_case(self, case_path: Path) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Preprocess a single case, using both KiTS23 ground truth and TotalSegmentor predictions.
        
        Returns:
            List of tuples (input_tensor, target_tensor) where:
            - input_tensor has shape (2, D, H, W) containing:
              - Channel 0: Normalized CT image
              - Channel 1: TotalSegmentor kidney mask
            - target_tensor has shape (1, D, H, W) containing tumor mask
        """
        # Load raw data
        image_chunks, kidney_mask_chunks, tumor_mask_chunks = self.load_case(case_path)
        
        if not image_chunks:
            print(f"Failed to load case {case_path.name}")
            return []
            
        # Get TotalSegmentor kidney mask
        try:
            img_path = case_path / "imaging.nii.gz"
            totalseg_kidney_mask = self.kidney_segmentor.get_kidney_mask(img_path, case_path.name)
            if totalseg_kidney_mask is None:
                print(f"Failed to generate TotalSegmentor mask for {case_path.name}")
                return []
                
            # Split TotalSegmentor mask into chunks matching our image chunks
            totalseg_chunks = []
            chunk_size = image_chunks[0].shape[0]
            for i in range(len(image_chunks)):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size
                if end_idx > totalseg_kidney_mask.shape[0]:
                    end_idx = totalseg_kidney_mask.shape[0]
                    chunk = totalseg_kidney_mask[start_idx:end_idx]
                    # Pad if necessary
                    if chunk.shape[0] < chunk_size:
                        padding = chunk_size - chunk.shape[0]
                        chunk = np.pad(chunk, ((0, padding), (0, 0), (0, 0)))
                else:
                    chunk = totalseg_kidney_mask[start_idx:end_idx]
                totalseg_chunks.append(chunk)
                
        except Exception as e:
            print(f"Error processing TotalSegmentor mask for {case_path.name}: {e}")
            return []
            
        return self.extract_patches(image_chunks, totalseg_chunks, tumor_mask_chunks)
        
    def load_case(self, case_path: Path) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Load case data in chunks without normalization."""
        img_path = case_path / "imaging.nii.gz"
        mask_path = case_path / "segmentation.nii.gz"
            
        if not img_path.exists() or not mask_path.exists():
            print(f"Could not find imaging or segmentation files for {case_path.name}")
            return [], [], []

        try:
            # Load data
            img_obj = nib.load(str(img_path))
            mask_obj = nib.load(str(mask_path))
            shape = img_obj.shape

            # Process in chunks
            chunk_size = min(shape[0], max(32, self.patch_size[0]))
            num_chunks = (shape[0] + chunk_size - 1) // chunk_size

            image_chunks = []
            mask_chunks = []
            tumor_chunks = []

            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, shape[0])
                
                # Load chunks
                img_chunk = np.asarray(img_obj.dataobj[start_idx:end_idx], dtype=self.processing_dtype)
                mask_chunk = np.asarray(mask_obj.dataobj[start_idx:end_idx], dtype=self.processing_dtype)

                # Get tumor mask (value 2)
                tumor_chunk = (mask_chunk == 2).astype(self.processing_dtype)

                image_chunks.append(img_chunk)
                mask_chunks.append(mask_chunk)
                tumor_chunks.append(tumor_chunk)

            return image_chunks, mask_chunks, tumor_chunks

        except Exception as e:
            print(f"Failed during load_case for {case_path.name}: {e}")
            return [], [], []
        finally:
            del img_obj
            del mask_obj
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def extract_patches(self, image_chunks: List[np.ndarray], 
                       kidney_mask_chunks: List[np.ndarray],
                       tumor_mask_chunks: List[np.ndarray]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Extract patches with global normalization."""
        if not image_chunks or not kidney_mask_chunks or not tumor_mask_chunks:
            return []

        # First concatenate and normalize entire volume
        image_volume = np.concatenate(image_chunks, axis=0)
        volume_min = np.min(image_volume)
        volume_max = np.max(image_volume)
        if volume_max > volume_min:
            image_volume = (image_volume - volume_min) / (volume_max - volume_min)
        else:
            image_volume = np.zeros_like(image_volume)

        # Split back into chunks for patch extraction
        chunk_sizes = [chunk.shape[0] for chunk in image_chunks]
        start_idx = 0
        normalized_chunks = []
        for size in chunk_sizes:
            normalized_chunks.append(image_volume[start_idx:start_idx + size])
            start_idx += size

        patches = []
        h_range = normalized_chunks[0].shape[1] - self.patch_size[1] + 1
        w_range = normalized_chunks[0].shape[2] - self.patch_size[2] + 1

        for img_chunk, kidney_chunk, tumor_chunk in zip(normalized_chunks, 
                                                      kidney_mask_chunks,
                                                      tumor_mask_chunks):
            for d in range(0, img_chunk.shape[0] - self.patch_size[0] + 1, self.stride[0]):
                for h in range(0, h_range, self.stride[1]):
                    for w in range(0, w_range, self.stride[2]):
                        img_patch = img_chunk[d:d+self.patch_size[0],
                                           h:h+self.patch_size[1],
                                           w:w+self.patch_size[2]]
                        kidney_patch = kidney_chunk[d:d+self.patch_size[0],
                                                h:h+self.patch_size[1],
                                                w:w+self.patch_size[2]]
                        tumor_patch = tumor_chunk[d:d+self.patch_size[0],
                                              h:h+self.patch_size[1],
                                              w:w+self.patch_size[2]]

                        if tumor_patch.sum() > 0:  # Only keep patches with tumor annotations
                            input_patch = np.stack([img_patch, kidney_patch], axis=0)
                            input_tensor = torch.from_numpy(input_patch).float()
                            tumor_mask_tensor = torch.from_numpy(tumor_patch).float().unsqueeze(0)
                            patches.append((input_tensor, tumor_mask_tensor))

        return patches