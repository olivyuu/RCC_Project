import torch
import numpy as np
from gradcam_3d import MultiScaleGradCAM
from typing import Tuple, Optional

class FullVolumeInference:
    def __init__(self, model, config=None, debug=False):
        self.model = model
        self.debug = debug
        
        # Initialize multi-scale GradCAM++
        target_layers = MultiScaleGradCAM.get_target_layers_from_model(model)
        weights = MultiScaleGradCAM.get_default_layer_weights()
        self.gradcam = MultiScaleGradCAM(
            model=model,
            target_layers=target_layers,
            weights=weights,
            use_cuda=torch.cuda.is_available()
        )
        
        if self.debug:
            print("\nInitialized MultiScaleGradCAM with layers:")
            for layer, weight in zip(target_layers, weights):
                print(f"  {layer.__class__.__name__}: weight={weight:.2f}")
    
    def sliding_window_inference(self, 
                               volume: np.ndarray,
                               kidney_mask: Optional[np.ndarray] = None,
                               window_size: Tuple[int, int, int] = (64, 128, 128),
                               overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference on full volume using sliding window
        
        Args:
            volume: Input volume (D, H, W)
            kidney_mask: Optional kidney segmentation mask
            window_size: Size of sliding window
            overlap: Overlap between windows
            
        Returns:
            predictions: Prediction probabilities (C, D, H, W)
            attention_maps: Combined GradCAM++ attention maps (D, H, W)
        """
        # Prepare input
        volume = torch.from_numpy(volume).float()
        if len(volume.shape) == 3:
            volume = volume.unsqueeze(0)  # Add channel dim
        if torch.cuda.is_available():
            volume = volume.cuda()
            
        # Get sliding window parameters
        stride = [int(s * (1 - overlap)) for s in window_size]
        padding = [s - (v % s) for s, v in zip(stride, volume.shape[1:])]
        
        # Pad volume
        volume = torch.nn.functional.pad(volume, (0, padding[2], 0, padding[1], 0, padding[0]))
        
        # Initialize output tensors
        output_shape = (2,) + volume.shape[1:]  # (C, D, H, W)
        predictions = torch.zeros(output_shape)
        counts = torch.zeros(output_shape)
        attention_maps = torch.zeros(volume.shape[1:])  # (D, H, W)
        attention_counts = torch.zeros(volume.shape[1:])
        
        if torch.cuda.is_available():
            predictions = predictions.cuda()
            counts = counts.cuda()
            attention_maps = attention_maps.cuda()
            attention_counts = attention_counts.cuda()
            
        # Sliding window inference
        D, H, W = volume.shape[1:]
        d_windows = range(0, D - window_size[0] + 1, stride[0])
        h_windows = range(0, H - window_size[1] + 1, stride[1])
        w_windows = range(0, W - window_size[2] + 1, stride[2])
        
        total_windows = len(d_windows) * len(h_windows) * len(w_windows)
        if self.debug:
            from tqdm import tqdm
            print("\nStarting sliding window inference with Grad-CAM...")
            windows = tqdm(total=total_windows)
        
        for d in d_windows:
            for h in h_windows:
                for w in w_windows:
                    # Extract window
                    d_slice = slice(d, d + window_size[0])
                    h_slice = slice(h, h + window_size[1])
                    w_slice = slice(w, w + window_size[2])
                    
                    window = volume[:, d_slice, h_slice, w_slice]
                    
                    # Run inference
                    with torch.cuda.amp.autocast(enabled=True):
                        with torch.no_grad():
                            pred = self.model(window)
                            
                    # Generate GradCAM++ attention
                    attention = self.gradcam(window, target_category=1)  # Focus on tumor class
                    attention = torch.from_numpy(attention)
                    if torch.cuda.is_available():
                        attention = attention.cuda()
                    
                    # Accumulate predictions and attention
                    predictions[:, d_slice, h_slice, w_slice] += pred[0]
                    counts[:, d_slice, h_slice, w_slice] += 1
                    attention_maps[d_slice, h_slice, w_slice] += attention
                    attention_counts[d_slice, h_slice, w_slice] += 1
                    
                    if self.debug:
                        windows.update(1)
        
        if self.debug:
            windows.close()
            
        # Average predictions and attention
        predictions = predictions / counts
        attention_maps = attention_maps / attention_counts
        
        # Apply kidney mask if provided
        if kidney_mask is not None:
            if self.debug:
                print("Applying kidney mask...")
            kidney_mask = torch.from_numpy(kidney_mask)
            if torch.cuda.is_available():
                kidney_mask = kidney_mask.cuda()
            predictions = predictions * kidney_mask.unsqueeze(0)
            attention_maps = attention_maps * kidney_mask
            
        # Convert to numpy
        predictions = predictions.cpu().numpy()
        attention_maps = attention_maps.cpu().numpy()
        
        # Unpad
        predictions = predictions[:, :D-padding[0], :H-padding[1], :W-padding[2]]
        attention_maps = attention_maps[:D-padding[0], :H-padding[1], :W-padding[2]]
        
        return predictions, attention_maps
    
    def _find_high_prob_slices(self, predictions: np.ndarray, axis: int = 0,
                              n_slices: int = 3, threshold: float = 0.5) -> list:
        """Find slices with highest tumor probabilities"""
        # Get tumor probabilities
        tumor_probs = predictions[1]  # Class 1 is tumor
        
        # Calculate mean probability for each slice
        if axis == 0:
            slice_probs = np.mean(tumor_probs, axis=(1,2))
        elif axis == 1:
            slice_probs = np.mean(tumor_probs, axis=(0,2))
        else:
            slice_probs = np.mean(tumor_probs, axis=(0,1))
            
        # Find slices above threshold
        high_prob_slices = np.where(slice_probs > threshold)[0]
        
        # If none found, take highest n_slices
        if len(high_prob_slices) == 0:
            high_prob_slices = np.argsort(slice_probs)[-n_slices:]
            
        # Sort by probability
        high_prob_slices = sorted(high_prob_slices, key=lambda i: slice_probs[i], reverse=True)
        
        return high_prob_slices[:n_slices]
