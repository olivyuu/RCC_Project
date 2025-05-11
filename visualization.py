import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import os
import scipy.ndimage as ndimage
from mpl_toolkits.mplot3d import Axes3D

def get_tumor_regions(output, threshold=0.5):
    """
    Find slices that contain tumor predictions
    
    Args:
        output: Model output tensor [1, 2, D, H, W]
        threshold: Probability threshold for tumor prediction
        
    Returns:
        list of slice indices with tumor predictions
    """
    tumor_probs = torch.softmax(output, dim=1)[0, 1]  # Get tumor class probabilities
    tumor_mask = tumor_probs > threshold
    tumor_slices = []
    
    # Find regions with tumor predictions
    regions = []
    for slice_idx in range(tumor_mask.shape[0]):
        if tumor_mask[slice_idx].any():
            # Get connected components in this slice
            labeled, num_components = ndimage.label(tumor_mask[slice_idx].cpu().numpy())
            
            for component in range(1, num_components + 1):
                component_mask = labeled == component
                y, x = np.where(component_mask)
                if len(y) > 0:  # If component is not empty
                    # Get bounding box with padding
                    min_y, max_y = np.min(y), np.max(y)
                    min_x, max_x = np.min(x), np.max(x)
                    
                    # Add padding (20% of region size)
                    height = max_y - min_y
                    width = max_x - min_x
                    pad_y = int(height * 0.2)
                    pad_x = int(width * 0.2)
                    
                    # Ensure bounds are within image
                    min_y = max(0, min_y - pad_y)
                    max_y = min(tumor_mask.shape[1], max_y + pad_y)
                    min_x = max(0, min_x - pad_x)
                    max_x = min(tumor_mask.shape[2], max_x + pad_x)
                    
                    regions.append({
                        'slice': slice_idx,
                        'bbox': (min_y, max_y, min_x, max_x),
                        'confidence': float(torch.mean(tumor_probs[slice_idx, min_y:max_y, min_x:max_x]))
                    })
    
    return regions

def generate_gradcam(model, input_tensor, target_class=1):
    """
    Generate Grad-CAM visualization for the model's predictions
    
    Args:
        model: The neural network model
        input_tensor: Input image tensor [1, 1, D, H, W]
        target_class: Index of the target class (default: 1 for tumor)
    
    Returns:
        cam: Grad-CAM heatmap
        pred: Model prediction
    """
    # Ensure model is in eval mode
    model.eval()
    
    # Forward pass
    input_tensor.requires_grad = True
    output = model(input_tensor)
    
    if isinstance(output, list):  # Handle deep supervision outputs
        output = output[0]
    
    # Get the score for target class
    if len(output.shape) == 5:  # If output includes batch dimension
        target_score = output[0, target_class]
    else:
        target_score = output[target_class]
    
    # Convert to scalar by taking mean over spatial dimensions
    target_score = target_score.mean()
    
    # Backward pass
    model.zero_grad()
    target_score.backward()
    
    # Get gradients and activations
    gradients = model.gradients[0]  # [C, D, H, W]
    activations = model.activations[0]  # [C, D, H, W]
    
    # Calculate importance weights
    weights = torch.mean(gradients, dim=(1, 2, 3))  # Global average pooling
    
    # Generate weighted activation map
    cam = torch.zeros_like(activations[0])  # Initialize CAM
    for i, w in enumerate(weights):
        cam += w * activations[i]
    
    # Apply ReLU and normalize
    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    
    return cam.detach().cpu().numpy(), output.detach().cpu()

def visualize_full_scan(image, cam, prediction, alpha=0.5, save_path=None):
    """
    Visualize Grad-CAM results for the full 3D scan with multi-planar reconstructions
    
    Args:
        image: Original 3D image [D, H, W]
        cam: Grad-CAM heatmap [D, H, W]
        prediction: Model prediction tensor
        alpha: Transparency of the heatmap
        save_path: Path to save the visualization
    """
    # Get tumor probabilities
    tumor_probs = torch.softmax(prediction, dim=1)[0, 1].cpu().numpy()
    
    # Create threshold mask
    thresh_mask = tumor_probs > 0.5
    
    # Get center of mass of tumor region
    if np.any(thresh_mask):
        center = ndimage.center_of_mass(thresh_mask)
    else:
        center = (image.shape[0]//2, image.shape[1]//2, image.shape[2]//2)
    
    # Create figure with multiple views
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(2, 3, figure=fig)
    
    # Axial view (top-down)
    ax1 = fig.add_subplot(gs[0, 0])
    slice_idx = int(center[0])
    ax1.imshow(image[slice_idx], cmap='gray')
    ax1.imshow(cam[slice_idx], cmap='jet', alpha=alpha)
    ax1.set_title(f'Axial View (Slice {slice_idx})')
    ax1.axis('off')
    
    # Sagittal view (side)
    ax2 = fig.add_subplot(gs[0, 1])
    slice_idx = int(center[2])
    sagittal_img = image[:, :, slice_idx]
    sagittal_cam = cam[:, :, slice_idx]
    ax2.imshow(sagittal_img, cmap='gray')
    ax2.imshow(sagittal_cam, cmap='jet', alpha=alpha)
    ax2.set_title(f'Sagittal View (Slice {slice_idx})')
    ax2.axis('off')
    
    # Coronal view (front)
    ax3 = fig.add_subplot(gs[0, 2])
    slice_idx = int(center[1])
    coronal_img = image[:, slice_idx, :]
    coronal_cam = cam[:, slice_idx, :]
    ax3.imshow(coronal_img, cmap='gray')
    ax3.imshow(coronal_cam, cmap='jet', alpha=alpha)
    ax3.set_title(f'Coronal View (Slice {slice_idx})')
    ax3.axis('off')
    
    # 3D visualization of tumor regions
    ax4 = fig.add_subplot(gs[1, 0], projection='3d')
    x, y, z = np.where(thresh_mask)
    if len(x) > 0:
        scatter = ax4.scatter(x, y, z, c=tumor_probs[x, y, z],
                            cmap='jet', alpha=0.6)
        plt.colorbar(scatter, ax=ax4, label='Tumor Probability')
    ax4.set_title('3D Tumor Visualization')
    
    # Probability distribution
    ax5 = fig.add_subplot(gs[1, 1])
    prob_hist = ax5.hist(tumor_probs[thresh_mask].flatten(), bins=50,
                        range=(0, 1), density=True)
    ax5.set_title('Tumor Probability Distribution')
    ax5.set_xlabel('Probability')
    ax5.set_ylabel('Density')
    
    # Summary statistics
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    stats_text = (
        f"Tumor Statistics:\n\n"
        f"Volume: {np.sum(thresh_mask)} voxels\n"
        f"Max Probability: {np.max(tumor_probs):.3f}\n"
        f"Mean Probability: {np.mean(tumor_probs[thresh_mask]):.3f}\n"
        f"Location: {center}\n"
    )
    ax6.text(0.1, 0.5, stats_text, fontsize=12, transform=ax6.transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def visualize_gradcam(image, cam, prediction, slice_indices=None, alpha=0.5, save_path=None):
    """
    Visualize Grad-CAM results in a clinically relevant format
    
    Args:
        image: Original 3D image [D, H, W]
        cam: Grad-CAM heatmap [1, H, W] or [D, H, W]
        prediction: Model prediction tensor
        slice_indices: List of slice indices to visualize (if None, shows all slices with tumors)
        alpha: Transparency of the heatmap
        save_path: Path to save the visualization (optional)
    """
    # Handle case where CAM is single slice
    if len(cam.shape) == 2 or (len(cam.shape) == 3 and cam.shape[0] == 1):
        cam = np.repeat(cam.reshape(1, *cam.shape[-2:]), image.shape[0], axis=0)
    
    # Get tumor regions
    regions = get_tumor_regions(prediction)
    
    if not regions:
        print("No tumor detected in any scan.")
        return
        
    # First create the full scan visualization
    if save_path:
        base, ext = os.path.splitext(save_path)
        full_scan_path = f"{base}_full_scan{ext}"
    else:
        full_scan_path = None
        
    visualize_full_scan(image, cam, prediction, alpha, full_scan_path)
    
    # Then create detailed views for each region
    for region_idx, region in enumerate(regions):
        slice_idx = region['slice']
        min_y, max_y, min_x, max_x = region['bbox']
        confidence = region['confidence']
        
        # Create figure with gridspec
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1])
        
        # Full slice view with tumor region highlighted
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image[slice_idx], cmap='gray')
        rect = Rectangle((min_x, min_y), max_x-min_x, max_y-min_y,
                        fill=False, edgecolor='red', linewidth=2)
        ax1.add_patch(rect)
        ax1.set_title(f'Full Slice View - Slice {slice_idx}')
        ax1.axis('off')
        
        # Zoomed original image
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(image[slice_idx, min_y:max_y, min_x:max_x], cmap='gray')
        ax2.set_title('Zoomed Region - Original')
        ax2.axis('off')
        
        # Zoomed Grad-CAM overlay
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(image[slice_idx, min_y:max_y, min_x:max_x], cmap='gray')
        heatmap = ax3.imshow(cam[slice_idx, min_y:max_y, min_x:max_x],
                            cmap='jet', alpha=alpha)
        ax3.set_title(f'Zoomed Region - Heatmap\nConfidence: {confidence:.2%}')
        ax3.axis('off')
        
        # Get tumor probabilities for current slice
        tumor_probs = torch.softmax(prediction, dim=1)[0, 1]  # [D, H, W]
        
        # Bottom row: Detailed analysis
        # Probability map
        ax4 = fig.add_subplot(gs[1, 0])
        prob_map = ax4.imshow(tumor_probs[slice_idx].cpu().numpy(), cmap='RdYlBu_r')
        ax4.set_title('Tumor Probability Map')
        ax4.axis('off')
        plt.colorbar(prob_map, ax=ax4, label='Tumor Probability')
        
        # Thresholded region
        ax5 = fig.add_subplot(gs[1, 1])
        thresh_map = (tumor_probs[slice_idx] > 0.5).float().cpu().numpy()
        ax5.imshow(image[slice_idx], cmap='gray')
        ax5.imshow(thresh_map, cmap='RdYlBu_r', alpha=0.5)
        ax5.set_title('Thresholded Tumor Region')
        ax5.axis('off')
        
        # Combined visualization
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.imshow(image[slice_idx], cmap='gray')
        
        # Create focused attention map
        overlay = np.zeros_like(cam[slice_idx])
        h, w = overlay.shape
        
        # Ensure bounds are within image dimensions
        min_y = max(0, min_y)
        max_y = min(h, max_y)
        min_x = max(0, min_x)
        max_x = min(w, max_x)
        
        # Extract region from threshold map and CAM
        region_thresh = thresh_map[min_y:max_y, min_x:max_x]
        region_cam = cam[slice_idx, min_y:max_y, min_x:max_x]
        
        # Only apply overlay in the region where threshold is True
        overlay[min_y:max_y, min_x:max_x] = np.where(region_thresh, region_cam, 0)
        heatmap = ax6.imshow(overlay, cmap='jet', alpha=alpha)
        ax6.set_title('Focused Attention Map')
        ax6.axis('off')
        plt.colorbar(heatmap, ax=ax6, label='Model Attention')
        
        plt.tight_layout()
        
        if save_path:
            # For multiple regions, save with different names
            base, ext = os.path.splitext(save_path)
            current_save_path = f"{base}_region{region_idx+1}{ext}"
            plt.savefig(current_save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()