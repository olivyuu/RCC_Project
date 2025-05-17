import torch
import matplotlib.pyplot as plt
from pathlib import Path
from model import nnUNetv2
from full_volume_inference import FullVolumeInference
from kidney_segmentor import KidneySegmentor
from config import nnUNetConfig
import random
import os
import numpy as np
import nibabel as nib
import argparse
from tqdm import tqdm
from skimage.feature import peak_local_max
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

# Set up matplotlib for high quality medical visualization
def load_model(checkpoint_path, debug=False):
    """Load the trained model from checkpoint"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if debug:
        print(f"\nLoading model from: {checkpoint_path}")
        print(f"Using device: {device}")
    # Initialize model
    config = nnUNetConfig()
    model = nnUNetv2(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        features=config.features
    )
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        if debug:
            print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise
    return model.to(device), device
def debug_model_output(model, dummy_input, checkpoint_path=None):
    """Debug model's forward pass"""
    print("\nDebugging model output:")
    try:
        # Check model architecture
        print("\nModel architecture:")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        if checkpoint_path:
            checkpoint_size = Path(checkpoint_path).stat().st_size / (1024 * 1024)
            print(f"Checkpoint size: {checkpoint_size:.2f}MB")
        # Test forward pass
        print("\nTesting forward pass:")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Input range: [{dummy_input.min():.2f}, {dummy_input.max():.2f}]")
        with torch.cuda.amp.autocast(enabled=True):
            output = model(dummy_input)
        if isinstance(output, list):
            print(f"Output is a list of {len(output)} tensors")
            for i, out in enumerate(output):
                print(f"Output[{i}] shape: {out.shape}")
                print(f"Output[{i}] range: [{out.min().item():.2f}, {out.max().item():.2f}]")
        else:
            print(f"Output shape: {output.shape}")
            print(f"Output range: [{output.min().item():.2f}, {output.max().item():.2f}]")
    except Exception as e:
        print(f"Error in model debugging: {str(e)}")
        raise

plt.style.use('dark_background')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [20, 15]  # Made wider for 3 panels

def visualize_results(img_slice, pred_slice, attn_slice, tumor_gt_slice, name, save_path, save_raw=False, spacing=(1.0, 1.0), debug=False):
    """Create and save clinically-relevant visualization of results"""
    try:
        if debug:
            print(f"\nInput shapes:")
            print(f"Image: {img_slice.shape}, Range: [{img_slice.min():.2f}, {img_slice.max():.2f}]")
            print(f"Predictions: {pred_slice.shape}, Range: [{pred_slice.min():.2f}, {pred_slice.max():.2f}]")
            print(f"Attention: {attn_slice.shape}, Range: [{attn_slice.min():.2f}, {attn_slice.max():.2f}]")
            if tumor_gt_slice is not None:
                print(f"Ground Truth: {tumor_gt_slice.shape}, Range: [{tumor_gt_slice.min():.2f}, {tumor_gt_slice.max():.2f}]")
            print(f"Spacing: {spacing}")
            
        # Create figure with three side-by-side panels
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
        fig.suptitle(f'Tumor Detection Analysis\n{name}', fontsize=16, y=0.95)
        
        # Left panel: Original CT scan
        ax1.imshow(img_slice, cmap='gray')
        ax1.set_title('Original CT Scan', fontsize=14)
        ax1.axis('off')
        
        # Add scale bar
        scalebar = AnchoredSizeBar(ax1.transData,
                                 100/spacing[0],  # 10mm in pixels
                                 '10 mm',
                                 'lower right',
                                 pad=0.5,
                                 color='white',
                                 frameon=False,
                                 size_vertical=1)
        ax1.add_artist(scalebar)
        
        # Middle panel: Ground truth overlay
        ax2.imshow(img_slice, cmap='gray')
        if tumor_gt_slice is not None:
            # Create tumor mask overlay
            tumor_mask = np.ma.masked_where(tumor_gt_slice == 0, tumor_gt_slice)
            ax2.imshow(tumor_mask, cmap='Reds', alpha=0.6)
        ax2.set_title('Ground Truth Tumor', fontsize=14)
        ax2.axis('off')
        
        # Right panel: Model predictions
        ax3.imshow(img_slice, cmap='gray')
        
        # Normalize attention map to [0,1]
        attn_norm = (attn_slice - attn_slice.min()) / (attn_slice.max() - attn_slice.min() + 1e-8)
        
        # Create weighted heatmap combining attention and probability
        combined_heatmap = attn_norm * pred_slice
        combined_heatmap = np.clip(combined_heatmap, 0, 1)
        
        # Display heatmap
        heatmap = ax3.imshow(combined_heatmap, 
                           cmap='RdYlBu_r',
                           alpha=0.7)
        ax3.set_title('Model Predictions', fontsize=14)
        ax3.axis('off')
        
        # Add colorbar with probability scale
        cbar = plt.colorbar(heatmap, ax=ax3)
        cbar.set_label('Tumor Probability Score', fontsize=12)
        
        # Add probability annotations for high confidence predictions
        try:
            high_prob_threshold = 0.75
            if debug:
                print("\nFinding peaks:")
                print(f"Combined heatmap range: [{combined_heatmap.min():.2f}, {combined_heatmap.max():.2f}]")
            
            # Ensure input is properly formatted for peak_local_max
            combined_heatmap = combined_heatmap.astype(np.float64)
            # Use absolute threshold instead of relative
            threshold_abs = high_prob_threshold * combined_heatmap.max()
            peak_coords = peak_local_max(
                combined_heatmap,
                min_distance=10,
                threshold_abs=threshold_abs,
                exclude_border=False
            )
            
            if debug:
                print(f"Input range for peak detection: [{combined_heatmap.min():.3f}, {combined_heatmap.max():.3f}]")
                print(f"Absolute threshold: {threshold_abs:.3f}")
                print(f"Found {len(peak_coords)} peaks above threshold {high_prob_threshold}")
                
            for y, x in peak_coords:
                if pred_slice[y, x] > high_prob_threshold:
                    ax3.text(x, y, f'{pred_slice[y, x]:.2f}',
                            color='white', fontsize=8,
                            ha='center', va='center',
                            bbox=dict(facecolor='black', alpha=0.5, pad=1))
                            
        except Exception as e:
            print(f"Warning: Peak detection failed: {str(e)}")
        
        # Save visualization
        plt.tight_layout()
        plt.savefig(save_path / 'analysis.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Save raw data if requested
        if save_raw:
            save_path.mkdir(exist_ok=True, parents=True)
            np.save(save_path / 'scan.npy', img_slice)
            np.save(save_path / 'probabilities.npy', pred_slice)
            np.save(save_path / 'heatmap.npy', combined_heatmap)
            if tumor_gt_slice is not None:
                np.save(save_path / 'ground_truth.npy', tumor_gt_slice)
            
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        raise

def process_full_volume(model, device, config, case_path, output_dir, debug=False, save_raw=False, spacing=None):
    """Process a full volume with Grad-CAM"""
    print(f"\nProcessing case: {case_path.name}")
    
    try:
        # Check available GPU memory
        if torch.cuda.is_available() and debug:
            print(f"\nInitial GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        
        # Initialize inference handler
        if debug:
            print("Initializing FullVolumeInference...")
        full_volume_handler = FullVolumeInference(model, config)
        
        # Initialize kidney segmentor
        if debug:
            print("Initializing KidneySegmentor...")
        kidney_segmentor = KidneySegmentor(debug=debug)
        
        # Load image with spacing information
        img_path = case_path / "imaging.nii.gz"
        if not img_path.exists():
            img_path = case_path / "raw_data" / "imaging.nii.gz"
        
        if not img_path.exists():
            raise FileNotFoundError(f"No imaging file found in {case_path}")
            
        try:
            img_obj = nib.load(str(img_path))
            volume = img_obj.get_fdata()
            if debug:
                print(f"Loaded volume shape: {volume.shape}")
                print(f"Volume range: [{volume.min():.2f}, {volume.max():.2f}]")
        except Exception as e:
            print(f"Error loading NIFTI file: {str(e)}")
            raise
            
        # Load tumor ground truth if available
        tumor_gt = None
        try:
            # Check both possible locations
            gt_paths = [
                case_path / "aggregated_MAJ_seg.nii.gz",
                case_path / "raw_data" / "aggregated_MAJ_seg.nii.gz"
            ]
            for gt_path in gt_paths:
                if gt_path.exists():
                    tumor_gt = nib.load(str(gt_path)).get_fdata()
                    # Convert to binary mask (tumor is label 2 in KiTS23)
                    tumor_gt = (tumor_gt == 2).astype(np.float32)
                    if debug:
                        print(f"Loaded tumor ground truth, shape: {tumor_gt.shape}")
                        print(f"Tumor voxels: {np.count_nonzero(tumor_gt)}")
                    break
        except Exception as e:
            print(f"Error loading ground truth: {str(e)}")
            tumor_gt = None
        
        # Get voxel spacing
        try:
            spacing = img_obj.header.get_zooms()
            if debug:
                print(f"Voxel spacing (mm): {spacing}")
        except:
            spacing = (1.0, 1.0, 1.0)
            
        # Get kidney segmentation
        if debug:
            print("Generating kidney segmentation...")
        kidney_mask = kidney_segmentor.get_kidney_mask(img_path, case_path.name)
        
        if kidney_mask is None:
            print("Warning: Failed to generate kidney mask, proceeding without anatomical constraints")
        elif debug:
            print(f"Kidney mask shape: {kidney_mask.shape}")
            print(f"Kidney mask range: [{kidney_mask.min():.2f}, {kidney_mask.max():.2f}]")
        
        # Run inference
        if debug:
            print("Running inference...")
        predictions, attention_maps = full_volume_handler.sliding_window_inference(volume, kidney_mask)
        
        if debug:
            print(f"Volume shape: {volume.shape}")
            print(f"Predictions shape: {predictions.shape}")
            print(f"Attention maps shape: {attention_maps.shape}")
        
        # Create output directory
        volume_dir = output_dir / case_path.name / 'full_volume'
        volume_dir.mkdir(exist_ok=True, parents=True)
        
        # Find slices with high tumor probabilities for each view
        slice_indices = {
            'Axial': full_volume_handler._find_high_prob_slices(predictions, axis=0),
            'Sagittal': full_volume_handler._find_high_prob_slices(predictions, axis=1),
            'Coronal': full_volume_handler._find_high_prob_slices(predictions, axis=2)
        }
        
        # Create visualizations for each view
        for axis, name in enumerate(['Axial', 'Sagittal', 'Coronal']):
            if debug:
                print(f"\nProcessing {name} view slices...")
                
            for slice_idx in slice_indices[name]:
                try:
                    # Extract appropriate slices
                    if axis == 0:  # Axial
                        img_slice = volume[slice_idx, :, :]
                        pred_slice = predictions[1, slice_idx, :, :]
                        attn_slice = attention_maps[slice_idx, :, :]
                        tumor_gt_slice = tumor_gt[slice_idx, :, :] if tumor_gt is not None else None
                        current_spacing = spacing[1:]
                    elif axis == 1:  # Sagittal
                        img_slice = volume[:, slice_idx, :]
                        pred_slice = predictions[1, :, slice_idx, :]
                        attn_slice = attention_maps[:, slice_idx, :]
                        tumor_gt_slice = tumor_gt[:, slice_idx, :] if tumor_gt is not None else None
                        current_spacing = (spacing[0], spacing[2])
                    else:  # Coronal
                        img_slice = volume[:, :, slice_idx]
                        pred_slice = predictions[1, :, :, slice_idx]
                        attn_slice = attention_maps[:, :, slice_idx]
                        tumor_gt_slice = tumor_gt[:, :, slice_idx] if tumor_gt is not None else None
                        current_spacing = spacing[:2]
                        
                    if debug:
                        print(f"\nSlice shapes for {name} view, slice {slice_idx}:")
                        print(f"Image: {img_slice.shape}")
                        print(f"Prediction: {pred_slice.shape}")
                        print(f"Attention: {attn_slice.shape}")
                        if tumor_gt_slice is not None:
                            print(f"Ground Truth: {tumor_gt_slice.shape}")
                        
                    # Create slice-specific directory
                    slice_dir = volume_dir / f'{name.lower()}_slice_{slice_idx}'
                    slice_dir.mkdir(exist_ok=True, parents=True)
                    
                    # Generate visualization
                    visualize_results(
                        img_slice, pred_slice, attn_slice, tumor_gt_slice,
                        f'{name} View - Slice {slice_idx}',
                        slice_dir, save_raw,
                        spacing=current_spacing,
                        debug=debug
                    )
                    
                except Exception as e:
                    print(f"Error processing slice {slice_idx}: {str(e)}")
                    if debug:
                        import traceback
                        print(traceback.format_exc())
                    continue
                
        if debug:
            print("Processing complete")
            
    except Exception as e:
        print(f"Error processing case {case_path.name}: {str(e)}")
        if debug:
            import traceback
            print(traceback.format_exc())
        raise

def main():
    parser = argparse.ArgumentParser(description='Generate full-volume Grad-CAM visualizations')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--cases', type=int, default=1,
                      help='Number of cases to process')
    parser.add_argument('--save_raw', action='store_true',
                      help='Save raw numpy arrays for further analysis')
    parser.add_argument('--output_dir', type=str, default='gradcam_results',
                      help='Directory to save results')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug output')
    
    args = parser.parse_args()
    
    try:
        # Load model
        print("\nInitializing...")
        model, device = load_model(args.checkpoint, args.debug)
        model.eval()
        
        if args.debug:
            # Test with dummy input
            dummy_input = torch.randn(1, 1, 64, 128, 128).to(device)
            debug_model_output(model, dummy_input, args.checkpoint)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Get case list and verify data
        config = nnUNetConfig()
        data_dir = Path(config.data_dir)
        
        if not data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")
            
        # Find cases with ground truth segmentations
        all_cases = []
        for case_dir in sorted(data_dir.iterdir()):
            if not case_dir.is_dir() or not case_dir.name.startswith('case_'):
                continue
                
            # Check for required files
            has_imaging = (case_dir / "imaging.nii.gz").exists() or \
                         (case_dir / "raw_data" / "imaging.nii.gz").exists()
                         
            has_seg = (case_dir / "aggregated_MAJ_seg.nii.gz").exists() or \
                     (case_dir / "raw_data" / "aggregated_MAJ_seg.nii.gz").exists()
                     
            if has_imaging and has_seg:
                all_cases.append(case_dir)
        
        if not all_cases:
            raise ValueError(f"No valid cases found in {data_dir}")
            
        if args.debug:
            print(f"\nData directory: {data_dir}")
            print(f"Total cases found: {len(all_cases)}")
            
        # Randomly select cases with ground truth
        selected_cases = random.sample(all_cases, min(args.cases, len(all_cases)))
        
        if args.debug:
            print(f"\nSelected cases: {[case.name for case in selected_cases]}")
            print(f"Running with {'debug enabled' if args.debug else 'debug disabled'}")
            
        # Process each case
        for i, case_path in enumerate(selected_cases, 1):
            print(f"\nProcessing case {i}/{len(selected_cases)}: {case_path.name}")
            try:
                process_full_volume(
                    model, device, config, case_path, output_dir,
                    debug=args.debug, save_raw=args.save_raw
                )
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error processing case {case_path.name}: {str(e)}")
                if args.debug:
                    raise
                continue
                
        print("\nProcessing complete!")
        
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    main()