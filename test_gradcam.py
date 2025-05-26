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
import skimage.feature
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import glob
import traceback
from tools.debug_utils import DebugLogger, validate_volume_shapes, gpu_memory_check, force_cuda_sync

# Set up matplotlib for high quality medical visualization
plt.style.use('dark_background')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [20, 15]  # Made wider for 3 panels

def load_model(checkpoint_path, debug=False, debug_logger=None):
    """Load the trained model from checkpoint"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if debug_logger:
        debug_logger.logger.info(f"\nLoading model from: {checkpoint_path}")
        debug_logger.logger.info(f"Using device: {device}")
    
    # Initialize model
    config = nnUNetConfig()
    model = nnUNetv2(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        features=config.features
    )
    
    try:
        with gpu_memory_check():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            if debug_logger:
                debug_logger.logger.info("Model loaded successfully")
                debug_logger.log_memory("After model load")
    except Exception as e:
        if debug_logger:
            debug_logger.logger.error(f"Error loading model: {str(e)}")
        raise
        
    return model.to(device), device

def debug_model_output(model, dummy_input, checkpoint_path=None, debug_logger=None):
    """Debug model's forward pass"""
    if debug_logger:
        debug_logger.logger.info("\nDebugging model output:")
    try:
        # Check model architecture
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if debug_logger:
            debug_logger.logger.info(f"Total parameters: {total_params:,}")
            debug_logger.logger.info(f"Trainable parameters: {trainable_params:,}")
        
        if checkpoint_path:
            checkpoint_size = Path(checkpoint_path).stat().st_size / (1024 * 1024)
            if debug_logger:
                debug_logger.logger.info(f"Checkpoint size: {checkpoint_size:.2f}MB")
        
        # Test forward pass
        if debug_logger:
            debug_logger.logger.info("\nTesting forward pass:")
            debug_logger.log_shapes("Input", input=dummy_input)
            debug_logger.log_stats("Input", input=dummy_input)
        
        with gpu_memory_check():
            with torch.cuda.amp.autocast(enabled=True):
                output = model(dummy_input)
        
        if isinstance(output, list):
            if debug_logger:
                debug_logger.logger.info(f"Output is a list of {len(output)} tensors")
                for i, out in enumerate(output):
                    debug_logger.log_shapes(f"Output[{i}]", output=out)
                    debug_logger.log_stats(f"Output[{i}]", output=out)
        else:
            if debug_logger:
                debug_logger.log_shapes("Output", output=output)
                debug_logger.log_stats("Output", output=output)
            
    except Exception as e:
        if debug_logger:
            debug_logger.logger.error(f"Error in model debugging: {str(e)}")
        raise

def visualize_results(img_slice, pred_slice, attn_slice, tumor_gt_slice, name, save_path, save_raw=False, spacing=(1.0, 1.0), debug=False, debug_logger=None):
    """Create and save clinically-relevant visualization of results"""
    try:
        if debug_logger:
            debug_logger.logger.debug(f"\nVisualizing {name}:")
            debug_logger.log_stats("Input arrays",
                                 image=torch.tensor(img_slice),
                                 pred=torch.tensor(pred_slice),
                                 attn=torch.tensor(attn_slice))
            if tumor_gt_slice is not None:
                debug_logger.log_stats("Ground Truth", gt=torch.tensor(tumor_gt_slice))
            
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

        if debug_logger:
            debug_logger.log_stats("Heatmap", 
                                 attn_norm=torch.tensor(attn_norm),
                                 combined=torch.tensor(combined_heatmap))
        
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
            if debug_logger:
                debug_logger.logger.debug("\nFinding peaks:")
                debug_logger.logger.debug(f"Combined heatmap range: [{combined_heatmap.min():.2f}, {combined_heatmap.max():.2f}]")
            
            # Ensure input is properly formatted for peak_local_max
            combined_heatmap = combined_heatmap.astype(np.float64)
            threshold_abs = high_prob_threshold * combined_heatmap.max()
            peak_coords = skimage.feature.peak_local_max(
                combined_heatmap,
                min_distance=10,
                threshold_abs=threshold_abs,
                exclude_border=False
            )
            
            if debug_logger:
                debug_logger.logger.debug(f"Found {len(peak_coords)} peaks above threshold {high_prob_threshold}")
                
            for y, x in peak_coords:
                if pred_slice[y, x] > high_prob_threshold:
                    ax3.text(x, y, f'{pred_slice[y, x]:.2f}',
                            color='white', fontsize=8,
                            ha='center', va='center',
                            bbox=dict(facecolor='black', alpha=0.5, pad=1))
                            
        except Exception as e:
            if debug_logger:
                debug_logger.logger.warning(f"Peak detection failed: {str(e)}")
        
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
        if debug_logger:
            debug_logger.logger.error(f"Error in visualization: {str(e)}")
        raise

def process_full_volume(model, device, case_path, output_dir, debug=False, save_raw=False, spacing=None, debug_logger=None):
    """Process a full volume with Grad-CAM"""
    if debug_logger:
        debug_logger.logger.info(f"\nProcessing case: {case_path.name}")
    
    try:
        # Check available GPU memory
        if torch.cuda.is_available() and debug_logger:
            debug_logger.log_memory("Initial state")
        
        # Initialize inference handler
        if debug_logger:
            debug_logger.logger.debug("Initializing FullVolumeInference...")
        full_volume_handler = FullVolumeInference(model, config=None)  # Config not needed for inference
        
        # Initialize kidney segmentor
        if debug_logger:
            debug_logger.logger.debug("Initializing KidneySegmentor...")
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
            if debug_logger:
                debug_logger.logger.debug(f"Loaded volume shape: {volume.shape}")
                debug_logger.log_stats("Volume", volume=torch.tensor(volume))
        except Exception as e:
            if debug_logger:
                debug_logger.logger.error(f"Error loading NIFTI file: {str(e)}")
            raise
            
        # Load tumor ground truth from instance files
        tumor_gt = None
        try:
            instances_dir = case_path / "instances"
            if instances_dir.exists():
                # Initialize empty tumor mask
                tumor_gt = np.zeros_like(volume)
                
                # Look for all tumor instance files
                for tumor_file in instances_dir.glob("tumor_instance-*_annotation-*.nii.gz"):
                    if debug_logger:
                        debug_logger.logger.debug(f"Loading tumor instance: {tumor_file.name}")
                    instance_mask = nib.load(str(tumor_file)).get_fdata()
                    # Add to combined mask (any non-zero value indicates tumor)
                    tumor_gt[instance_mask > 0] = 1
                    
                if debug_logger:
                    if np.any(tumor_gt):
                        debug_logger.logger.debug("Tumor ground truth loaded:")
                        debug_logger.log_stats("Ground Truth", gt=torch.tensor(tumor_gt))
                        tumor_locations = np.where(tumor_gt > 0)
                        debug_logger.logger.debug(
                            f"Tumor extent: z={min(tumor_locations[0])}-{max(tumor_locations[0])}, "
                            f"y={min(tumor_locations[1])}-{max(tumor_locations[1])}, "
                            f"x={min(tumor_locations[2])}-{max(tumor_locations[2])}"
                        )
                    else:
                        debug_logger.logger.debug("No tumor instances found in ground truth")
                        
        except Exception as e:
            if debug_logger:
                debug_logger.logger.error(f"Error loading ground truth: {str(e)}")
            tumor_gt = None
        
        # Get voxel spacing
        try:
            spacing = img_obj.header.get_zooms()
            if debug_logger:
                debug_logger.logger.debug(f"Voxel spacing (mm): {spacing}")
        except:
            spacing = (1.0, 1.0, 1.0)
            
        # Get kidney segmentation
        if debug_logger:
            debug_logger.logger.debug("Generating kidney segmentation...")
        kidney_mask = kidney_segmentor.get_kidney_mask(img_path, case_path.name)
        
        if kidney_mask is None:
            if debug_logger:
                debug_logger.logger.warning("Failed to generate kidney mask, proceeding without anatomical constraints")
        elif debug_logger:
            debug_logger.log_stats("Kidney Mask", mask=torch.tensor(kidney_mask))
        
        # Run inference
        if debug_logger:
            debug_logger.logger.debug("Running inference...")
            force_cuda_sync()  # Ensure clean memory state
            
        with gpu_memory_check():
            predictions, attention_maps = full_volume_handler.sliding_window_inference(volume, kidney_mask)
            force_cuda_sync()  # Ensure inference is complete
        
        if debug_logger:
            debug_logger.log_shapes("Inference results",
                                  volume=torch.tensor(volume),
                                  predictions=torch.tensor(predictions),
                                  attention=torch.tensor(attention_maps))
            debug_logger.log_stats("Predictions",
                                 predictions=torch.tensor(predictions),
                                 attention=torch.tensor(attention_maps))
        
        # Create output directory
        volume_dir = output_dir / case_path.name / 'full_volume'
        volume_dir.mkdir(exist_ok=True, parents=True)
        
        # Find slices with high tumor probabilities for each view
        slice_indices = {
            'Axial': full_volume_handler._find_high_prob_slices(predictions, axis=0),
            'Sagittal': full_volume_handler._find_high_prob_slices(predictions, axis=1),
            'Coronal': full_volume_handler._find_high_prob_slices(predictions, axis=2)
        }
        
        if debug_logger:
            debug_logger.logger.debug("\nHigh probability slice indices:")
            for view, indices in slice_indices.items():
                debug_logger.logger.debug(f"{view}: {indices}")
        
        # Create visualizations for each view
        for axis, name in enumerate(['Axial', 'Sagittal', 'Coronal']):
            if debug_logger:
                debug_logger.logger.debug(f"\nProcessing {name} view slices...")
                
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
                        
                    if debug_logger:
                        debug_logger.logger.debug(f"\nProcessing {name} view, slice {slice_idx}")
                        debug_logger.log_shapes("Slice data",
                                             image=torch.tensor(img_slice),
                                             pred=torch.tensor(pred_slice),
                                             attn=torch.tensor(attn_slice),
                                             gt=torch.tensor(tumor_gt_slice) if tumor_gt_slice is not None else None)
                        
                    # Create slice-specific directory
                    slice_dir = volume_dir / f'{name.lower()}_slice_{slice_idx}'
                    slice_dir.mkdir(exist_ok=True, parents=True)
                    
                    # Generate visualization
                    visualize_results(
                        img_slice, pred_slice, attn_slice, tumor_gt_slice,
                        f'{name} View - Slice {slice_idx}',
                        slice_dir, save_raw,
                        spacing=current_spacing,
                        debug=debug,
                        debug_logger=debug_logger
                    )
                    
                except Exception as e:
                    if debug_logger:
                        debug_logger.logger.error(f"Error processing slice {slice_idx}: {str(e)}")
                        debug_logger.logger.error(traceback.format_exc())
                    continue
                
        if debug_logger:
            debug_logger.logger.info("Processing complete")
            debug_logger.log_memory("Final state")
            
    except Exception as e:
        if debug_logger:
            debug_logger.logger.error(f"Error processing case {case_path.name}: {str(e)}")
            debug_logger.logger.error(traceback.format_exc())
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
        checkpoint_path = Path(args.checkpoint)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # Initialize debug logger
        debug_logger = DebugLogger(output_dir, debug=args.debug) if args.debug else None
        
        # Load model
        model, device = load_model(checkpoint_path, args.debug, debug_logger)
        model.eval()
        
        # Debug model
        if args.debug:
            dummy_input = torch.randn(1, 2, 128, 256, 256).to(device)
            try:
                validate_volume_shapes(dummy_input)
            except ValueError as e:
                if debug_logger:
                    debug_logger.logger.error(f"Volume shape validation failed: {str(e)}")
                raise
            debug_model_output(model, dummy_input, checkpoint_path, debug_logger)
        
        # Process cases
        kits_dir = Path("/workspace/kits23/dataset")
        all_cases = sorted([d for d in kits_dir.glob("case_*") if d.is_dir()])
        
        # Select random subset if needed
        if 0 < args.cases < len(all_cases):
            selected_cases = random.sample(all_cases, args.cases)
        else:
            selected_cases = all_cases[:args.cases]
            
        # Process each case
        for case_path in tqdm(selected_cases, desc="Processing cases"):
            process_full_volume(
                model, device, case_path, output_dir,
                debug=args.debug, save_raw=args.save_raw,
                debug_logger=debug_logger
            )
            
    except Exception as e:
        if debug_logger:
            debug_logger.logger.error(f"Error in main: {str(e)}")
            debug_logger.logger.error(traceback.format_exc())
        raise
        
    finally:
        if debug_logger:
            debug_logger.log_memory("Program end")

if __name__ == '__main__':
    main()