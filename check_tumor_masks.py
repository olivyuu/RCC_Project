import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

def analyze_tumor_masks_in_pt(data_dir: str):
    """Analyze tumor masks in preprocessed .pt volumes"""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        return

    print(f"Looking in directory: {data_dir}")
    all_files = sorted(os.listdir(data_dir))
    print("\nAll files in directory (first 10):")
    for f in all_files[:10]:
        print(f"  {f}")
    print(f"... and {len(all_files) - 10} more files\n")

    zero_masks = []
    pos_masks = []
    tumor_volumes = []
    error_files = []

    print("Analyzing tumor masks inside *.pt files...")
    for fname in tqdm(sorted(all_files)):
        if not fname.endswith(".pt"):
            continue

        pt_path = data_dir / fname
        try:
            data = torch.load(pt_path)
            
            # Print structure of first file to help debug
            if len(zero_masks) + len(pos_masks) == 0:
                print("\nFirst file structure:")
                print(f"Data type: {type(data)}")
                if isinstance(data, tuple):
                    print(f"Tuple length: {len(data)}")
                    for i, item in enumerate(data):
                        print(f"Item {i}: type={type(item)}, shape={item.shape if hasattr(item, 'shape') else 'no shape'}")
                elif isinstance(data, dict):
                    print("Dictionary keys:", list(data.keys()))
                    for k, v in data.items():
                        print(f"Key '{k}': type={type(v)}, shape={v.shape if hasattr(v, 'shape') else 'no shape'}")

            # Extract tumor mask based on data structure
            if isinstance(data, (tuple, list)):
                if len(data) == 2:  # (image, mask) format
                    _, tumor_mask = data
                elif len(data) == 3:  # (image, kidney_mask, tumor_mask) format
                    _, _, tumor_mask = data
                else:
                    print(f"\nUnexpected tuple length in {fname}: {len(data)}")
                    error_files.append(fname)
                    continue
            elif isinstance(data, dict):
                if 'tumor' in data:
                    tumor_mask = data['tumor']
                elif 'mask' in data:
                    tumor_mask = data['mask']
                else:
                    print(f"\nNo tumor mask found in {fname} keys: {list(data.keys())}")
                    error_files.append(fname)
                    continue
            else:
                print(f"\nUnexpected data type in {fname}: {type(data)}")
                error_files.append(fname)
                continue

            # Ensure tumor_mask is a tensor and on CPU
            if isinstance(tumor_mask, torch.Tensor):
                tumor_mask = tumor_mask.cpu()

            # Calculate tumor volume
            total_tumor_vox = float(tumor_mask.sum())
            max_value = float(tumor_mask.max())
            unique_values = torch.unique(tumor_mask).tolist()

            if total_tumor_vox == 0:
                zero_masks.append((fname, max_value, unique_values))
            else:
                pos_masks.append((fname, total_tumor_vox, max_value, unique_values))
                tumor_volumes.append(total_tumor_vox)

        except Exception as e:
            print(f"\nError processing {fname}: {str(e)}")
            error_files.append(fname)
            continue

    print("\nAnalysis Results:")
    print(f"Total .pt files checked: {len(all_files)}")
    print(f"Cases with tumor (mask sum > 0): {len(pos_masks)}")
    print(f"Cases without tumor (mask sum = 0): {len(zero_masks)}")
    print(f"Files with errors: {len(error_files)}")

    if tumor_volumes:
        print("\nTumor volume statistics (over positive cases):")
        print(f"Mean: {np.mean(tumor_volumes):.2f}")
        print(f"Median: {np.median(tumor_volumes):.2f}")
        print(f"Min: {np.min(tumor_volumes):.2f}")
        print(f"Max: {np.max(tumor_volumes):.2f}")

    if zero_masks:
        print("\nFirst few cases without tumor:")
        for fname, max_val, unique_vals in zero_masks[:5]:
            print(f"  {fname}")
            print(f"    Max value: {max_val}")
            print(f"    Unique values: {unique_vals}")

    if pos_masks:
        print("\nFirst few cases with tumor:")
        for fname, vol, max_val, unique_vals in pos_masks[:5]:
            print(f"  {fname}")
            print(f"    Total volume: {vol:.2f}")
            print(f"    Max value: {max_val}")
            print(f"    Unique values: {unique_vals}")

    if error_files:
        print("\nFiles with errors:")
        for fname in error_files:
            print(f"  {fname}")

if __name__ == "__main__":
    data_dir = os.getenv("PREPROCESSED_DIR", "preprocessed_volumes")
    analyze_tumor_masks_in_pt(data_dir)