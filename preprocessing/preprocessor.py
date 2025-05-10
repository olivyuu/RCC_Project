import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Tuple, List
import torch
from tqdm import tqdm
import gc
from nibabel.processing import resample_from_to  # Changed from resample_img

class KiTS23Preprocessor:
    def __init__(self, config):
        self.config = config
        self.patch_size = config.patch_size
        self.stride = config.patch_stride
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Use float32 for calculations
        self.processing_dtype = np.float32
        self.processing_dtype_itemsize = np.dtype(self.processing_dtype).itemsize
        # Default max estimated memory based on config file size limit * 1.5, assuming float32 processing
        # We might need to adjust this based on observed resampling behavior
        self.max_estimated_memory_mb = getattr(config, 'max_estimated_memory_mb', config.max_image_size_mb * 1.5)
        print(f"Preprocessor initialized with patch size: {self.patch_size}")
        print(f"Target processing dtype: {self.processing_dtype}")
        print(f"Max estimated memory limit (for {self.processing_dtype}): {self.max_estimated_memory_mb:.1f} MB")

    def load_case(self, case_path: Path) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Load and preprocess a case with size thresholds and memory management.
        Includes estimated memory check based on header info, attempts float32 conversion early.
        """
        print(f"\nLoading case: {case_path.name}")
        img_path = case_path / "raw_data" / "imaging.nii.gz"
        if not img_path.exists():
            img_path = case_path / "imaging.nii.gz"  # Try fallback path
        
        mask_path = case_path / "raw_data" / "segmentation.nii.gz"
        if not mask_path.exists():
            mask_path = case_path / "segmentation.nii.gz"  # Try fallback path
            
        if not img_path.exists() or not mask_path.exists():
            print(f"Could not find imaging or segmentation files for {case_path.name}")
            print(f"Expected paths:")
            print(f"  {img_path} or {case_path}/imaging.nii.gz")
            print(f"  {mask_path} or {case_path}/segmentation.nii.gz")
            print("\nData installation required:")
            print("1. Install the KiTS23 package:")
            print("   git clone https://github.com/neheller/kits23")
            print("   cd kits23")
            print("   pip3 install -e .")
            print("\n2. Download the dataset:")
            print("   kits23_download_data")
            print("\nThis will place the data in the dataset/ folder.")
            print("Expected files for each case:")
            print(f"  - {img_path}")
            print(f"  - {mask_path}")
            print("\nFor more information, visit:")
            print("https://github.com/neheller/kits23#installation")
            return [], []
        img_obj = None # Initialize to ensure cleanup in finally block
        mask_obj = None

        try:
            # Log image size for monitoring
            img_size_mb = img_path.stat().st_size / (1024 * 1024)
            print(f"Loading case {case_path.name} ({img_size_mb:.1f} MB)")

            # 2. Load Headers (Minimal Memory)
            img_obj = nib.load(str(img_path))
            mask_obj = nib.load(str(mask_path))

            # Get shape and data info for logging
            shape = img_obj.shape
            original_dtype = img_obj.header.get_data_dtype()
            print(f"Image shape: {shape}, Original dtype: {original_dtype}")

            # 4. Dimension Check & Optional Downsampling
            needs_downsampling = any(s > m for s, m in zip(shape, self.config.max_image_dimensions))
            
            if needs_downsampling:
                if self.config.downsample_large_images:
                    print(f"Downsampling {case_path.name} due to large dimensions {shape}")
                    factor = self.config.downsample_factor
                    new_shape = tuple(max(1, s // factor) for s in shape)
                    
                    # Check estimated size *after* downsampling potential (using processing dtype)
                    downsampled_estimated_mb = np.prod(new_shape) * self.processing_dtype_itemsize / (1024 * 1024)
                    print(f"Potential downsampled shape: {new_shape}, Estimated size ({self.processing_dtype}): {downsampled_estimated_mb:.1f}MB")
                    
                    if downsampled_estimated_mb > self.max_estimated_memory_mb:
                         print(f"Skipping {case_path.name}: Estimated memory even after downsampling ({downsampled_estimated_mb:.1f}MB) exceeds limit {self.max_estimated_memory_mb:.1f}MB")
                         return [], [] # Cleanup happens in finally

                    # Create target reference image with new dimensions AND TARGET DTYPE
                    target_affine = img_obj.affine.copy()
                    # Create a dummy array with the target shape and dtype for the target image
                    target_data = np.zeros(new_shape, dtype=self.processing_dtype)
                    target_img = nib.Nifti1Image(target_data, target_affine)
                    # Clear the dummy data array
                    target_data = None
                    gc.collect()

                    print("Performing resampling (attempting float32 output)...")
                    img_obj_resampled = None # Initialize for cleanup
                    mask_obj_resampled = None
                    try:
                        # Resample image (order=1) - dtype argument removed as it's not supported
                        img_obj_resampled = resample_from_to(img_obj, target_img, order=1)
                        # Resample mask (order=0) - dtype argument removed
                        mask_obj_resampled = resample_from_to(mask_obj, target_img, order=0)
                        
                        # Explicitly delete original objects BEFORE assigning resampled ones
                        del img_obj
                        del mask_obj
                        gc.collect() # Collect garbage after deleting large objects
                        if torch.cuda.is_available(): torch.cuda.empty_cache()

                        img_obj = img_obj_resampled
                        mask_obj = mask_obj_resampled
                        shape = img_obj.shape # Use the new shape
                        print(f"New shape after downsampling: {shape}, New dtype: {img_obj.header.get_data_dtype()}")
                        
                    except Exception as e:
                        print(f"Error during resampling for {case_path.name}: {e}. Skipping case.")
                        # Cleanup intermediate objects if they exist
                        del img_obj_resampled
                        del mask_obj_resampled
                        gc.collect()
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                        return [], [] # Main cleanup in finally
                    finally:
                        # Ensure intermediate objects are deleted even if assignment failed before error
                        img_obj_resampled = None
                        mask_obj_resampled = None
                        target_img = None # Delete target reference image
                        gc.collect()

                else: # Not downsampling large images
                    print(f"Skipping {case_path.name}: Dimensions {shape} exceed limits and downsampling is disabled.")
                    return [], [] # Cleanup happens in finally

            # 5. Chunking (Now on original or downsampled data, hopefully float32 if downsampled)
            target_chunk_size = min(32, self.config.chunk_size)
            chunk_size = min(shape[0], max(target_chunk_size, self.patch_size[0]))
            if chunk_size <= 0:
                 print(f"Error: Invalid chunk size calculation for {case_path.name}. Shape: {shape}. Skipping.")
                 return [], [] # Cleanup happens in finally
            num_chunks = (shape[0] + chunk_size - 1) // chunk_size
            print(f"Processing in {num_chunks} chunks of size ~{chunk_size} along axis 0.")

            image_chunks = []
            mask_chunks = []

            for i in range(num_chunks):
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                gc.collect()

                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, shape[0])
                chunk_shape = (end_idx - start_idx,) + shape[1:]

                # Check estimated size of the chunk (using processing dtype)
                chunk_estimated_mb = np.prod(chunk_shape) * self.processing_dtype_itemsize / (1024 * 1024)
                # print(f"Chunk {i}: Indices [{start_idx}:{end_idx}], Shape {chunk_shape}, Estimated size: {chunk_estimated_mb:.1f}MB")
                if chunk_estimated_mb > self.max_estimated_memory_mb:
                    print(f"Warning: Estimated memory for chunk {i} ({chunk_estimated_mb:.1f}MB) exceeds limit {self.max_estimated_memory_mb:.1f}MB. Skipping chunk.")
                    continue

                img_chunk = None # Initialize for cleanup within loop
                mask_chunk = None
                img_chunk_normalized = None
                try:
                    # Load chunk using dataobj proxy AND convert to target dtype immediately
                    img_chunk = np.asarray(img_obj.dataobj[start_idx:end_idx], dtype=self.processing_dtype)
                    # Also convert mask chunk - float32 is fine for masks here
                    mask_chunk = np.asarray(mask_obj.dataobj[start_idx:end_idx], dtype=self.processing_dtype)

                    # Padding
                    actual_chunk_depth = img_chunk.shape[0]
                    if actual_chunk_depth < self.patch_size[0]:
                        pad_size = self.patch_size[0] - actual_chunk_depth
                        # Using 'constant' padding might be slightly less memory intensive than 'reflect'
                        img_chunk = np.pad(img_chunk, ((0, pad_size), (0, 0), (0, 0)), mode='constant', constant_values=0)
                        mask_chunk = np.pad(mask_chunk, ((0, pad_size), (0, 0), (0, 0)), mode='constant', constant_values=0)

                    # Normalization
                    chunk_min = np.min(img_chunk)
                    chunk_max = np.max(img_chunk)
                    if chunk_max > chunk_min:
                         # Perform normalization out-of-place
                         img_chunk_normalized = (img_chunk - chunk_min) / (chunk_max - chunk_min + 1e-8)
                    else:
                         img_chunk_normalized = np.zeros_like(img_chunk)

                    image_chunks.append(img_chunk_normalized)
                    mask_chunks.append(mask_chunk) # Append mask chunk (already padded)

                except (MemoryError, RuntimeError) as e:  # RuntimeError can occur for numpy memory issues
                    print(f"Memory error loading/processing chunk {i} for {case_path.name}, indices [{start_idx}:{end_idx}], shape {chunk_shape}. Skipping chunk: {e}")
                    # Explicitly clear potentially large variables from the failed try block
                    del img_chunk
                    del mask_chunk
                    del img_chunk_normalized
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    continue # Try next chunk
                except Exception as e:
                    print(f"Error processing chunk {i} for {case_path.name}: {e}")
                    del img_chunk
                    del mask_chunk
                    del img_chunk_normalized
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    continue # Try next chunk
                finally:
                    # Ensure intermediate arrays are deleted after processing or error
                    img_chunk = None
                    mask_chunk = None
                    img_chunk_normalized = None
                    # gc.collect() # Avoid collecting in loop unless necessary

            print(f"Successfully processed {len(image_chunks)} chunks for {case_path.name}.")
            return image_chunks, mask_chunks

        except Exception as e:
            print(f"Failed during load_case for {case_path.name}: {e}")
            return [], [] # Cleanup happens in finally
        finally:
            # Final cleanup of nibabel objects
            del img_obj
            del mask_obj
            img_obj = None
            mask_obj = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- extract_patches method ---
    # (No major changes needed here based on logs, but ensure it handles empty lists)
    def extract_patches(self, image_chunks: List[np.ndarray], mask_chunks: List[np.ndarray]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        patches = []
        total_patches_considered = 0
        total_empty_patches = 0

        if not image_chunks or not mask_chunks or len(image_chunks) != len(mask_chunks):
            print("Warning: Invalid or empty image/mask chunks provided to extract_patches.")
            return []

        # Get dimensions from the first chunk (assuming consistency)
        # Check if image_chunks[0] exists before accessing shape
        if not image_chunks[0].size > 0:
             print("Warning: First image chunk is empty.")
             return []
        chunk_D_example, H, W = image_chunks[0].shape # Use example shape, actual D varies per chunk
        print(f"Starting patch extraction with example chunk D={chunk_D_example}, H={H}, W={W}")

        # Basic dimension checks
        if H < self.patch_size[1] or W < self.patch_size[2]:
            print(f"Warning: Image H/W dimensions ({H},{W}) are smaller than patch H/W {self.patch_size[1:]}. Cannot extract patches.")
            return []

        h_range = H - self.patch_size[1] + 1
        w_range = W - self.patch_size[2] + 1

        pbar = tqdm(total=None, desc="Extracting patches", unit="patch", smoothing=0.1)
        
        img_chunk = None # Define outside loop for cleanup
        mask_chunk = None

        try: # Wrap the loop for better cleanup
            for i, (img_chunk, mask_chunk) in enumerate(zip(image_chunks, mask_chunks)):
                chunk_D = img_chunk.shape[0]

                if chunk_D < self.patch_size[0]:
                    continue

                chunk_d_range = chunk_D - self.patch_size[0] + 1

                for d_local in range(0, chunk_d_range, self.stride[0]):
                    for h in range(0, h_range, self.stride[1]):
                        for w in range(0, w_range, self.stride[2]):
                            total_patches_considered += 1

                            img_patch_view = None # Initialize for cleanup
                            mask_patch_view = None
                            img_patch_tensor = None
                            mask_patch_tensor = None
                            try:
                                img_patch_view = img_chunk[d_local:d_local+self.patch_size[0],
                                                     h:h+self.patch_size[1],
                                                     w:w+self.patch_size[2]]
                                mask_patch_view = mask_chunk[d_local:d_local+self.patch_size[0],
                                                       h:h+self.patch_size[1],
                                                       w:w+self.patch_size[2]]

                                if mask_patch_view.sum() < 1e-6:
                                    total_empty_patches += 1
                                    continue

                                # Convert to tensors only for valid patches
                                # Using .copy() for safety, as views might be invalidated later
                                img_patch_tensor = torch.from_numpy(img_patch_view.copy()).float().unsqueeze(0)
                                mask_patch_tensor = torch.from_numpy(mask_patch_view.copy()).float().unsqueeze(0)
                                
                                patches.append((img_patch_tensor, mask_patch_tensor))
                                pbar.update(1)

                            except MemoryError as e:
                                print(f"\nMemory error converting patch to tensor: {e}")
                                print(f"Patch shapes - Image: {img_patch_view.shape if img_patch_view is not None else 'N/A'}, Mask: {mask_patch_view.shape if mask_patch_view is not None else 'N/A'}")
                                # Explicit cleanup within exception handler
                                del img_patch_view
                                del mask_patch_view
                                del img_patch_tensor
                                del mask_patch_tensor
                                gc.collect()
                                # Maybe skip the rest of this chunk if memory is very low? Or just continue.
                                continue # Skip this patch
                            except Exception as e:
                                print(f"\nError converting patch to tensor: {e}")
                                del img_patch_view
                                del mask_patch_view
                                del img_patch_tensor
                                del mask_patch_tensor
                                gc.collect()
                                continue # Skip this patch
                            finally:
                                # Clear views/tensors after use or error
                                img_patch_view = None
                                mask_patch_view = None
                                img_patch_tensor = None
                                mask_patch_tensor = None
                                
                # Clear chunk data after processing all its patches
                img_chunk = None
                mask_chunk = None
                # gc.collect() # Avoid frequent collection

        finally: # Ensure cleanup even if loop breaks
            pbar.close()
            # Clear last chunk references if loop exited early
            img_chunk = None
            mask_chunk = None
            # Clear the input lists which hold references to the chunks
            image_chunks.clear()
            mask_chunks.clear()
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        # Print summary
        print(f"\nPatch extraction summary:")
        print(f"Total patches considered: {total_patches_considered}")
        print(f"Empty patches skipped: {total_empty_patches}")
        print(f"Valid patches extracted: {len(patches)}")

        return patches

    # --- preprocess_case method ---
    # (Ensure it calls the updated load_case and extract_patches)
    def preprocess_case(self, case_path: Path) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        image_chunks, mask_chunks = self.load_case(case_path)

        if not image_chunks:
             print(f"Warning: No image chunks loaded for {case_path.name}. Skipping patch extraction.")
             # Ensure cleanup even if loading fails
             image_chunks = []
             mask_chunks = []
             gc.collect()
             if torch.cuda.is_available(): torch.cuda.empty_cache()
             return []

        # Extract patches - cleanup is now handled inside extract_patches
        patches = self.extract_patches(image_chunks, mask_chunks)

        # Final check and cleanup
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        print(f"Extracted {len(patches)} patches from {case_path.name}")
        min_patches = getattr(self.config, 'min_patches_per_image', 1)
        if len(patches) < min_patches:
            print(f"Skipping {case_path.name}: Extracted {len(patches)} patches, less than minimum {min_patches}.")
            # Clear patches list explicitly before returning empty
            patches.clear()
            gc.collect()
            return []
        else:
            return patches