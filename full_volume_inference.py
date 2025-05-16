import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import nibabel as nib
from tqdm import tqdm

class FullVolumeInference:
    def __init__(self, model, config, debug=False):
        self.model = model
        self.config = config
        self.debug = debug
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.window_size = (64, 128, 128)
        self.overlap = 0.25
        
        # Set up target layer for Grad-CAM
        self.target_layer = self._setup_target_layer()
        self.activations = []
        self.gradients = []
        
    def _setup_target_layer(self):
        """Set up the target layer for Grad-CAM and register hooks"""
        target_layer = None
        
        # First try to find the deepest convolutional layer in the encoder
        if hasattr(self.model, 'encoder'):
            for module in reversed(self.model.encoder):
                if isinstance(module, torch.nn.Conv3d):
                    target_layer = module
                    break
        
        # If not found in encoder, look for bottleneck
        if target_layer is None:
            for name, module in self.model.named_modules():
                if 'bottleneck' in name.lower() and isinstance(module, torch.nn.Conv3d):
                    target_layer = module
                    break
        
        if target_layer is None:
            raise ValueError("Could not find suitable convolutional target layer for Grad-CAM")
            
        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)
        
        if self.debug:
            print(f"Registered Grad-CAM hooks on layer: {target_layer}")
            
        return target_layer
        
    def _save_activation(self, module, input, output):
        """Hook to save layer activations"""
        self.activations = [output.detach()]
        if self.debug:
            print(f"Activation shape: {output.shape}")
            print(f"Activation range: [{output.min().item():.2f}, {output.max().item():.2f}]")
        
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save layer gradients"""
        self.gradients = [grad_output[0].detach()]
        if self.debug:
            print(f"Gradient shape: {grad_output[0].shape}")
            print(f"Gradient range: [{grad_output[0].min().item():.2f}, {grad_output[0].max().item():.2f}]")
            if (grad_output[0] == 0).all():
                print("Warning: All gradients are zero!")
        
    def _compute_stride(self):
        """Compute stride based on window size and overlap"""
        return [int(ws * (1 - self.overlap)) for ws in self.window_size]
        
    def _preprocess_window(self, window):
        """Prepare a window tensor for model input"""
        window_tensor = torch.from_numpy(window).float()
        window_tensor = window_tensor.unsqueeze(0).unsqueeze(0)
        return window_tensor.to(self.device)
        
    def preprocess_volume(self, volume):
        """Preprocess the input volume"""
        volume = volume.astype(np.float32)
        volume = np.clip(volume, -1024, 3071)
        volume = (volume - (-1024)) / (3071 - (-1024))
        volume = volume * 2 - 1
        return volume
        
    def _pad_volume(self, volume):
        """Pad volume to handle window extraction"""
        d, h, w = volume.shape
        
        pad_d = (self.window_size[0] - d % self.window_size[0]) % self.window_size[0]
        pad_h = (self.window_size[1] - h % self.window_size[1]) % self.window_size[1]
        pad_w = (self.window_size[2] - w % self.window_size[2]) % self.window_size[2]
        
        padded = np.pad(volume,
                       ((0, pad_d), (0, pad_h), (0, pad_w)),
                       mode='constant')
        
        return padded, (pad_d, pad_h, pad_w)
        
    def _compute_gradcam(self, acts, grads, target_size):
        """Compute Grad-CAM attention map"""
        if acts is None or grads is None or len(acts) == 0 or len(grads) == 0:
            if self.debug:
                print("Missing activations or gradients")
            return None
            
        acts, grads = acts[0], grads[0]
        
        if self.debug:
            print(f"\nComputing GradCAM:")
            print(f"Activations shape: {acts.shape}")
            print(f"Activations range: [{acts.min().item():.2f}, {acts.max().item():.2f}]")
            print(f"Gradients shape: {grads.shape}")
            print(f"Gradients range: [{grads.min().item():.2f}, {grads.max().item():.2f}]")
            print(f"Target size: {target_size}")
        
        try:
            # Keep batch dimension for proper broadcasting
            weights = grads.mean(dim=(2, 3, 4), keepdim=True)  # Shape: (1, C, 1, 1, 1)
            
            if self.debug:
                print(f"Weights shape: {weights.shape}")
                print(f"Weights range: [{weights.min().item():.2f}, {weights.max().item():.2f}]")
            
            # Check for zero weights
            if (weights == 0).all():
                if self.debug:
                    print("Warning: All weights are zero!")
                return None
            
            # Compute weighted activations
            weighted_acts = weights * acts  # Shape: (1, C, D, H, W)
            cam = weighted_acts.sum(dim=1)  # Shape: (1, D, H, W)
            
            if self.debug:
                print(f"Weighted activations shape: {weighted_acts.shape}")
                print(f"Initial CAM shape: {cam.shape}")
            
            # Apply ReLU
            cam = F.relu(cam)
            
            if torch.isnan(cam).any():
                print("Warning: NaN values in CAM")
                return None
            
            # Normalize
            cam_min = cam.min()
            cam_max = cam.max()
            if cam_min == cam_max:
                if self.debug:
                    print("Warning: Constant CAM values")
                return None
            
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
            
            # Upsample to target size
            cam = F.interpolate(
                cam.unsqueeze(0),  # Add channel dim for interpolate: (1, 1, D, H, W)
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            
            if self.debug:
                print(f"Final CAM shape: {cam.shape}")
                print(f"Final CAM range: [{cam.min().item():.2f}, {cam.max().item():.2f}]")
            
            # Remove extra dimensions and convert to numpy
            return cam.squeeze().cpu().numpy()
            
        except Exception as e:
            if self.debug:
                print(f"Error in GradCAM computation: {str(e)}")
                import traceback
                print(traceback.format_exc())
            return None
            
    def _find_high_prob_slices(self, predictions, axis, num_slices=3, prob_threshold=0.9):
        """Find slices with highest tumor probabilities along given axis"""
        tumor_probs = predictions[1]  # Get tumor probability channel
        
        # Compute mean probability for each slice along the specified axis
        slice_probs = np.mean(tumor_probs, axis=tuple(i for i in range(tumor_probs.ndim) if i != axis))
        
        # Find indices of slices with probabilities above threshold
        high_prob_indices = np.where(slice_probs > prob_threshold)[0]
        
        if len(high_prob_indices) == 0:
            # If no slices above threshold, take slices with highest probabilities
            sorted_indices = np.argsort(slice_probs)
            selected_indices = sorted_indices[-num_slices:]
        else:
            # Sort high probability indices by their probability values
            sorted_high_prob = sorted(high_prob_indices, key=lambda i: slice_probs[i], reverse=True)
            selected_indices = sorted_high_prob[:num_slices]
        
        # Sort indices for consistent visualization
        selected_indices = sorted(selected_indices)
        
        if self.debug:
            print(f"\nSlice selection for axis {axis}:")
            print(f"Mean probabilities shape: {slice_probs.shape}")
            print(f"Selected indices: {selected_indices}")
            print(f"Corresponding probabilities: {[slice_probs[i] for i in selected_indices]}")
            
        return selected_indices
        
    def sliding_window_inference(self, volume):
        """Perform sliding window inference with Grad-CAM"""
        print("\nStarting sliding window inference with Grad-CAM...")
        
        # Preprocessing
        volume = self.preprocess_volume(volume)
        padded_volume, padding = self._pad_volume(volume)
        d, h, w = padded_volume.shape
        stride = self._compute_stride()
        
        # Initialize output arrays
        predictions = np.zeros((2,) + padded_volume.shape, dtype=np.float32)
        attention_maps = np.zeros_like(padded_volume)
        count_map = np.zeros_like(padded_volume)
        
        # Sliding window inference
        total_windows = ((d-1)//stride[0] + 1) * ((h-1)//stride[1] + 1) * ((w-1)//stride[2] + 1)
        with tqdm(total=total_windows) as pbar:
            for z in range(0, d - self.window_size[0] + 1, stride[0]):
                for y in range(0, h - self.window_size[1] + 1, stride[1]):
                    for x in range(0, w - self.window_size[2] + 1, stride[2]):
                        try:
                            # Extract and process window
                            window = padded_volume[
                                z:z + self.window_size[0],
                                y:y + self.window_size[1],
                                x:x + self.window_size[2]
                            ]
                            window_tensor = self._preprocess_window(window)
                            
                            if self.debug:
                                print(f"\nProcessing window at ({z},{y},{x})")
                                print(f"Window shape: {window.shape}")
                                print(f"Window tensor shape: {window_tensor.shape}")
                            
                            # Forward pass and predictions
                            output = self.model(window_tensor)
                            
                            # Get probabilities
                            with torch.no_grad():
                                probs = F.softmax(output, dim=1)[0, 1]
                                if probs.max().item() < 0.5:
                                    continue
                                    
                                # Store predictions
                                pred_window = self._compute_predictions(output, self.window_size)
                                predictions[:, z:z + self.window_size[0],
                                             y:y + self.window_size[1],
                                             x:x + self.window_size[2]] += pred_window
                                             
                            # Compute Grad-CAM
                            self.model.zero_grad()
                            target_score = output[0, 1].max()
                            
                            # Only compute GradCAM if prediction score is significant
                            if target_score > 0.5:
                                # Compute gradients
                                target_score.backward(retain_graph=True)
                                
                                # Generate attention map
                                attention = self._compute_gradcam(
                                    self.activations,
                                    self.gradients,
                                    self.window_size
                                )
                                
                                if attention is not None:
                                    attention_maps[z:z + self.window_size[0],
                                                y:y + self.window_size[1],
                                                x:x + self.window_size[2]] += attention
                                    count_map[z:z + self.window_size[0],
                                            y:y + self.window_size[1],
                                            x:x + self.window_size[2]] += 1
                                            
                            # Clear cache
                            self.activations = []
                            self.gradients = []
                            del window_tensor, output
                            torch.cuda.empty_cache()
                            
                        except Exception as e:
                            print(f"Error processing window at ({z},{y},{x}): {str(e)}")
                            if self.debug:
                                import traceback
                                print(traceback.format_exc())
                            continue
                            
                        pbar.update(1)
        
        # Post-process results
        predictions = predictions / np.maximum(count_map, 1)
        attention_maps = attention_maps / np.maximum(count_map, 1)
        
        # Remove padding
        pad_d, pad_h, pad_w = padding
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            predictions = predictions[:, :-pad_d if pad_d > 0 else None,
                                   :-pad_h if pad_h > 0 else None,
                                   :-pad_w if pad_w > 0 else None]
            attention_maps = attention_maps[:-pad_d if pad_d > 0 else None,
                                         :-pad_h if pad_h > 0 else None,
                                         :-pad_w if pad_w > 0 else None]
                                         
        return predictions, attention_maps
