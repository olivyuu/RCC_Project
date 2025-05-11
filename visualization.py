import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import os
import scipy.ndimage as ndimage

def get_tumor_regions(output, threshold=0.5):
    """Find slices that contain tumor predictions"""
    tumor_probs = torch.softmax(output, dim=1)[0, 1]  # Get tumor class probabilities
    tumor_mask = tumor_probs > threshold
    
    # Find regions with tumor predictions
    regions = []
    for slice_idx in range(tumor_mask.shape[0]):
        if tumor_mask[slice_idx].any():
            # Get connected components in this slice
            labeled, num_components = ndimage.label(tumor_mask[slice_idx].cpu().detach().numpy())
            
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
                        'confidence': float(torch.mean(tumor_probs[slice_idx, min_y:max_y, min_x:max_x]).detach())
                    })
    
    return regions

def generate_gradcam(model, input_tensor, target_class=1):
    """Generate Grad-CAM visualization"""
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
    """Visualize Grad-CAM results for patches"""
    # Ensure prediction is detached
    prediction = prediction.detach()
    
    # Get tumor probabilities
    tumor_probs = torch.softmax(prediction, dim=1)[0, 1].cpu().numpy()
    
    # Create threshold mask
    thresh_mask = tumor_probs > 0.5
    
    # Create figure with multiple views
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # Central slice for each view
    d, h, w = image.shape
    d_mid, h_mid, w_mid = d//2, h//2, w//2
    
    # Axial view (top-down)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image[d_mid], cmap='gray')
    ax1.imshow(cam[d_mid], cmap='jet', alpha=alpha)
    ax1.set_title(f'Axial View (Slice {d_mid})')
    ax1.axis('off')
    
    # Sagittal view (side)
    ax2 = fig.add_subplot(gs[0, 1])
    sagittal_img = image[:, :, w_mid]
    sagittal_cam = cam[:, :, w_mid]
    ax2.imshow(sagittal_img, cmap='gray')
    ax2.imshow(sagittal_cam, cmap='jet', alpha=alpha)
    ax2.set_title(f'Sagittal View (Central Slice)')
    ax2.axis('off')
    
    # Coronal view (front)
    ax3 = fig.add_subplot(gs[0, 2])
    coronal_img = image[:, h_mid, :]
    coronal_cam = cam[:, h_mid, :]
    ax3.imshow(coronal_img, cmap='gray')
    ax3.imshow(coronal_cam, cmap='jet', alpha=alpha)
    ax3.set_title(f'Coronal View (Central Slice)')
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
    if np.any(thresh_mask):
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
        f"Mean Probability: {np.mean(tumor_probs):.3f}\n"
        f"Patch Size: {image.shape}\n"
    )
    ax6.text(0.1, 0.5, stats_text, fontsize=12, transform=ax6.transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def visualize_gradcam(image, cam, prediction, slice_indices=None, alpha=0.5, save_path=None):
    """Visualize Grad-CAM results"""
    # Ensure prediction is detached
    prediction = prediction.detach()
    
    # First create the full scan visualization
    if save_path:
        base, ext = os.path.splitext(save_path)
        full_scan_path = f"{base}_full{ext}"
    else:
        full_scan_path = None
    
    visualize_full_scan(image, cam, prediction, alpha, full_scan_path)
    
    # Get tumor probabilities
    tumor_probs = torch.softmax(prediction, dim=1)[0, 1].cpu().numpy()
    
    # Create detailed view of central slice
    d_mid = image.shape[0] // 2
    
    # Create figure with gridspec
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1])
    
    # Full slice view
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image[d_mid], cmap='gray')
    ax1.set_title(f'Central Slice (Slice {d_mid})')
    ax1.axis('off')
    
    # Grad-CAM overlay
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(image[d_mid], cmap='gray')
    heatmap = ax2.imshow(cam[d_mid], cmap='jet', alpha=alpha)
    ax2.set_title('Grad-CAM Heatmap')
    ax2.axis('off')
    
    # Probability map
    ax3 = fig.add_subplot(gs[0, 2])
    prob_map = ax3.imshow(tumor_probs[d_mid], cmap='RdYlBu_r')
    ax3.set_title('Tumor Probability Map')
    ax3.axis('off')
    plt.colorbar(prob_map, ax=ax3, label='Tumor Probability')
    
    # Thresholded region
    ax4 = fig.add_subplot(gs[1, 0])
    thresh_map = tumor_probs[d_mid] > 0.5
    ax4.imshow(image[d_mid], cmap='gray')
    ax4.imshow(thresh_map, cmap='RdYlBu_r', alpha=0.5)
    ax4.set_title('Thresholded Tumor Region')
    ax4.axis('off')
    
    # Combined visualization
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(image[d_mid], cmap='gray')
    overlay = np.where(thresh_map, cam[d_mid], 0)
    heatmap = ax5.imshow(overlay, cmap='jet', alpha=alpha)
    ax5.set_title('Focused Attention Map')
    ax5.axis('off')
    plt.colorbar(heatmap, ax=ax5, label='Model Attention')
    
    # Summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    stats_text = (
        f"Slice Statistics:\n\n"
        f"Active Voxels: {np.sum(thresh_map)}\n"
        f"Max Probability: {np.max(tumor_probs[d_mid]):.3f}\n"
        f"Mean Probability: {np.mean(tumor_probs[d_mid]):.3f}\n"
        f"Patch Size: {image.shape}\n"
    )
    ax6.text(0.1, 0.5, stats_text, fontsize=12, transform=ax6.transAxes)
    
    plt.tight_layout()
    
    if save_path:
        base, ext = os.path.splitext(save_path)
        detail_path = f"{base}_detail{ext}"
        plt.savefig(detail_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()