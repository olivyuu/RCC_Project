import torch, cv2
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import os
import scipy.ndimage as ndimage
import traceback

def print_tensor_info(name, tensor):
    """Helper function to print tensor information"""
    if isinstance(tensor, torch.Tensor):
        print(f"{name}:")
        print(f"  Shape: {tensor.shape}")
        print(f"  Device: {tensor.device}")
        print(f"  Dtype: {tensor.dtype}")
        print(f"  Range: [{tensor.min().item():.3f}, {tensor.max().item():.3f}]")
        print(f"  Mean: {tensor.mean().item():.3f}")
        print(f"  Requires grad: {tensor.requires_grad}")
    elif isinstance(tensor, np.ndarray):
        print(f"{name}:")
        print(f"  Shape: {tensor.shape}")
        print(f"  Dtype: {tensor.dtype}")
        print(f"  Range: [{tensor.min():.3f}, {tensor.max():.3f}]")
        print(f"  Mean: {tensor.mean():.3f}")

def get_center_indices(shape):
    """Get central indices for each dimension"""
    centers = [dim // 2 for dim in shape]
    print(f"\nCalculating center indices for shape {shape}:")
    print(f"Center indices: {centers}")
    return centers

def get_prediction_center_indices(shape, prediction_shape):
    """
    Get central indices accounting for different input and prediction sizes
    """
    d, h, w = [dim // 2 for dim in shape]
    
    d_scale = prediction_shape[0] / shape[0]
    h_scale = prediction_shape[1] / shape[1]
    w_scale = prediction_shape[2] / shape[2]
    
    d_mid = min(int(d * d_scale), prediction_shape[0] - 1)
    h_mid = min(int(h * h_scale), prediction_shape[1] - 1)
    w_mid = min(int(w * w_scale), prediction_shape[2] - 1)
    
    print(f"\nScaling center indices:")
    print(f"Input shape: {shape}, centers: [{d}, {h}, {w}]")
    print(f"Prediction shape: {prediction_shape}, scaled centers: [{d_mid}, {h_mid}, {w_mid}]")
    
    return d_mid, h_mid, w_mid

def generate_gradcam(model, input_tensor, target_class=1):
    """Generate Grad-CAM visualization"""
    try:
        print("\nStarting Grad-CAM generation...")
        print_tensor_info("Input tensor", input_tensor)
        
        model.eval()
        
        # Forward pass
        input_tensor.requires_grad = True
        print("\nPerforming forward pass...")
        output = model(input_tensor)
        
        if isinstance(output, list):
            output = output[0]
        print_tensor_info("Model output", output)
        
        # Get the score for target class
        if len(output.shape) == 5:
            target_score = output[0, target_class]
        else:
            target_score = output[target_class]
        
        target_score = target_score.mean()
        print(f"\nTarget score: {target_score.item():.3f}")
        
        # Backward pass
        print("\nPerforming backward pass...")
        model.zero_grad()
        target_score.backward()
        
        # Get gradients and activations
        gradients = model.gradients[0]
        activations = model.activations[0]
        print_tensor_info("Gradients", gradients)
        print_tensor_info("Activations", activations)
        
        # Calculate importance weights
        weights = torch.mean(gradients, dim=(1, 2, 3))
        print(f"\nGradient weights shape: {weights.shape}")
        print(f"Weight range: [{weights.min().item():.3f}, {weights.max().item():.3f}]")
        
        # Generate weighted activation map
        cam = torch.zeros_like(activations[0])
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU and normalize
        cam = torch.relu(cam)
        print_tensor_info("\nInitial CAM", cam)
        
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        print_tensor_info("Normalized CAM", cam)
        
        # Upsample CAM to match input size
        cam = cam.unsqueeze(0).unsqueeze(0)
        print("\nUpsample dimensions:")
        print(f"  Input shape: {input_tensor.shape}")
        print(f"  Target shape: {input_tensor.shape[2:]}")
        print(f"  CAM shape before: {cam.shape}")
        
        cam = F.interpolate(
            cam, 
            size=input_tensor.shape[2:],
            mode='trilinear',
            align_corners=False
        )
        cam = cam.squeeze()
        print(f"  CAM shape after: {cam.shape}")
        print_tensor_info("Final upsampled CAM", cam)
        
        return cam.detach().cpu().numpy(), output.detach().cpu()
    
    except Exception as e:
        print("\nError in generate_gradcam:")
        print(f"Exception: {str(e)}")
        print("Stack trace:")
        traceback.print_exc()
        raise

def visualize_full_scan(image, cam, prediction, alpha=0.5, save_path=None):
    """Visualize Grad-CAM results for patches"""
    try:
        print("\nStarting full scan visualization...")
        print_tensor_info("Image", image)
        print_tensor_info("CAM", cam)
        print_tensor_info("Prediction", prediction)
        
        # Get tumor probabilities
        prediction = prediction.detach()
        tumor_probs = torch.softmax(prediction, dim=1)[0, 1].cpu().numpy()
        print_tensor_info("Tumor probabilities", tumor_probs)
        
        # Create threshold mask
        thresh_mask = tumor_probs > 0.5
        print(f"Threshold mask: {thresh_mask.sum()} positive voxels")
        
        # Create figure with multiple views
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, 3, figure=fig)
        
        # Get scaled center indices
        d_mid, h_mid, w_mid = get_prediction_center_indices(image.shape, tumor_probs.shape)
        print(f"Using scaled central slices: d={d_mid}, h={h_mid}, w={w_mid}")
        
        # Axial view (top-down)
        print("Creating axial view...")
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image[d_mid*2], cmap='gray')
        ax1.imshow(cam[d_mid*2], cmap='jet', alpha=alpha)
        ax1.set_title(f'Axial View (Slice {d_mid*2})')
        ax1.axis('off')
        
        # Sagittal view (side)
        print("Creating sagittal view...")
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(image[:, :, w_mid*2], cmap='gray')
        ax2.imshow(cam[:, :, w_mid*2], cmap='jet', alpha=alpha)
        ax2.set_title(f'Sagittal View (Slice {w_mid*2})')
        ax2.axis('off')
        
        # Coronal view (front)
        print("Creating coronal view...")
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(image[:, h_mid*2, :], cmap='gray')
        ax3.imshow(cam[:, h_mid*2, :], cmap='jet', alpha=alpha)
        ax3.set_title(f'Coronal View (Slice {h_mid*2})')
        ax3.axis('off')
        
        # 3D visualization of tumor regions
        print("Creating 3D visualization...")
        ax4 = fig.add_subplot(gs[1, 0], projection='3d')
        x, y, z = np.where(thresh_mask)
        if len(x) > 0:
            scatter = ax4.scatter(x, y, z, c=tumor_probs[x, y, z],
                                cmap='jet', alpha=0.6)
            plt.colorbar(scatter, ax=ax4, label='Tumor Probability')
        ax4.set_title('3D Tumor Visualization')
        
        # Probability distribution
        print("Creating probability distribution...")
        ax5 = fig.add_subplot(gs[1, 1])
        if np.any(thresh_mask):
            prob_hist = ax5.hist(tumor_probs[thresh_mask].flatten(), bins=50,
                                range=(0, 1), density=True)
        ax5.set_title('Tumor Probability Distribution')
        ax5.set_xlabel('Probability')
        ax5.set_ylabel('Density')
        
        # Summary statistics
        print("Adding summary statistics...")
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        stats_text = (
            f"Tumor Statistics:\n\n"
            f"Volume: {np.sum(thresh_mask)} voxels\n"
            f"Max Probability: {np.max(tumor_probs):.3f}\n"
            f"Mean Probability: {np.mean(tumor_probs):.3f}\n"
            f"Input Size: {image.shape}\n"
            f"Prediction Size: {tumor_probs.shape}\n"
        )
        ax6.text(0.1, 0.5, stats_text, fontsize=12, transform=ax6.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            print(f"Saving visualization to {save_path}")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            print("Visualization saved successfully")
        else:
            plt.show()
            print("Visualization displayed")
        
    except Exception as e:
        print("\nError in visualize_full_scan:")
        print(f"Exception: {str(e)}")
        print("Stack trace:")
        traceback.print_exc()
        raise

def visualize_gradcam(image, cam, prediction, slice_indices=None, alpha=0.5, save_path=None):
    """Visualize Grad-CAM results"""
    try:
        print("\nStarting Grad-CAM visualization...")
        print_tensor_info("Input image", image)
        print_tensor_info("CAM", cam)
        print_tensor_info("Prediction", prediction)
        
        # First create the full scan visualization
        if save_path:
            base, ext = os.path.splitext(save_path)
            full_scan_path = f"{base}_full{ext}"
            print(f"\nGenerating full scan visualization: {full_scan_path}")
        else:
            full_scan_path = None
            print("\nGenerating full scan visualization (display only)")
        
        visualize_full_scan(image, cam, prediction, alpha, full_scan_path)
        
        # Get tumor probabilities
        prediction = prediction.detach()
        tumor_probs = torch.softmax(prediction, dim=1)[0, 1].cpu().numpy()
        print_tensor_info("\nTumor probabilities for detailed view", tumor_probs)
        
        # Get scaled center indices
        d_mid, h_mid, w_mid = get_prediction_center_indices(image.shape, tumor_probs.shape)
        print(f"\nUsing scaled central slice {d_mid} for detailed view")
        
        # Create figure with gridspec
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1])
        
        # Full slice view
        print("Creating full slice view...")
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image[d_mid*2], cmap='gray')
        ax1.set_title(f'Central Slice (Slice {d_mid*2})')
        ax1.axis('off')
        
        # Grad-CAM overlay
        print("Creating Grad-CAM overlay...")
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(image[d_mid*2], cmap='gray')
        heatmap = ax2.imshow(cam[d_mid*2], cmap='jet', alpha=alpha)
        ax2.set_title('Grad-CAM Heatmap')
        ax2.axis('off')
        
        # Probability map
        print("Creating probability map...")
        ax3 = fig.add_subplot(gs[0, 2])
        prob_map = ax3.imshow(tumor_probs[d_mid], cmap='RdYlBu_r')
        ax3.set_title('Tumor Probability Map')
        ax3.axis('off')
        plt.colorbar(prob_map, ax=ax3, label='Tumor Probability')
        
        # Thresholded region
        print("Creating thresholded region view...")
        ax4 = fig.add_subplot(gs[1, 0])
        thresh_map = tumor_probs[d_mid] > 0.5
        ax4.imshow(image[d_mid*2], cmap='gray')
        ax4.imshow(thresh_map, cmap='RdYlBu_r', alpha=0.5)
        ax4.set_title('Thresholded Tumor Region')
        ax4.axis('off')
        
        # Combined visualization
        print("Creating combined visualization...")
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.imshow(image[d_mid*2], cmap='gray')
        # Debug shapes before resizing
        print("\nResizing debug info:")
        print(f"thresh_map shape: {thresh_map.shape}")
        print(f"thresh_map dtype: {thresh_map.dtype}")
        print(f"thresh_map range: [{thresh_map.min():.3f}, {thresh_map.max():.3f}]")
        print(f"target CAM shape: {cam[d_mid*2].shape}")
        print(f"target CAM range: [{cam[d_mid*2].min():.3f}, {cam[d_mid*2].max():.3f}]")
        
        # Resize thresh_map to match CAM dimensions
        thresh_map_resized = cv2.resize(
            thresh_map.astype(np.float32),
            (cam[d_mid*2].shape[1], cam[d_mid*2].shape[0]),
            interpolation=cv2.INTER_NEAREST)
        
        # Debug shapes after resizing
        print(f"Resized thresh_map shape: {thresh_map_resized.shape}")
        print(f"Resized thresh_map range: [{thresh_map_resized.min():.3f}, {thresh_map_resized.max():.3f}]")
        
        overlay = np.where(thresh_map_resized, cam[d_mid*2], 0)
        print(f"Final overlay shape: {overlay.shape}")
        print(f"Final overlay range: [{overlay.min():.3f}, {overlay.max():.3f}]\n")
        heatmap = ax5.imshow(overlay, cmap='jet', alpha=alpha)
        ax5.set_title('Focused Attention Map')
        ax5.axis('off')
        plt.colorbar(heatmap, ax=ax5, label='Model Attention')
        
        # Summary
        print("Adding detailed statistics...")
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        stats_text = (
            f"Slice Statistics:\n\n"
            f"Active Voxels: {np.sum(thresh_map)}\n"
            f"Max Probability: {np.max(tumor_probs[d_mid]):.3f}\n"
            f"Mean Probability: {np.mean(tumor_probs[d_mid]):.3f}\n"
            f"Input Size: {image.shape}\n"
            f"Prediction Size: {tumor_probs.shape}\n"
        )
        ax6.text(0.1, 0.5, stats_text, fontsize=12, transform=ax6.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            base, ext = os.path.splitext(save_path)
            detail_path = f"{base}_detail{ext}"
            print(f"\nSaving detailed visualization: {detail_path}")
            plt.savefig(detail_path, bbox_inches='tight', dpi=300)
            plt.close()
            print("Detailed visualization saved successfully")
        else:
            plt.show()
            print("Detailed visualization displayed")
        
    except Exception as e:
        print("\nError in visualize_gradcam:")
        print(f"Exception: {str(e)}")
        print("Stack trace:")
        traceback.print_exc()
        raise