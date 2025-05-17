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
        
        # Set up multiple target layers for Grad-CAM
        self.target_layers = self._setup_target_layers()
        self.activations = {name: [] for name in self.target_layers.keys()}
        self.gradients = {name: [] for name in self.target_layers.keys()}
        
    def _setup_target_layers(self):
        """Set up multiple target layers for Grad-CAM"""
        target_layers = {}
        
        # Get the deepest convolutional layers from each level
        if hasattr(self.model, 'down_blocks'):
            for i, block in enumerate(self.model.down_blocks):
                if hasattr(block, 'conv_block'):
                    target_layers[f'encoder_{i}'] = block.conv_block[-1].conv
        
        # Add bottleneck
        if hasattr(self.model, 'bottleneck'):
            target_layers['bottleneck'] = self.model.bottleneck.conv
            
        if not target_layers:
            raise ValueError("Could not find suitable target layers for Grad-CAM")
        
        # Register hooks for each layer
        for name, layer in target_layers.items():
            layer.register_forward_hook(lambda m, i, o, name=name: self._save_activation(m, i, o, name))
            layer.register_full_backward_hook(lambda m, i, o, name=name: self._save_gradient(m, i, o, name))
            
        if self.debug:
            print(f"Registered Grad-CAM hooks on layers: {list(target_layers.keys())}")
            
        return target_layers
        
    def _save_activation(self, module, input, output, name):
        """Hook to save layer activations"""
        self.activations[name] = [output.detach()]
        if self.debug:
            print(f"Activation {name} shape: {output.shape}")
            print(f"Activation {name} range: [{output.min().item():.2f}, {output.max().item():.2f}]")
        
    def _save_gradient(self, module, grad_input, grad_output, name):
        """Hook to save layer gradients"""
        self.gradients[name] = [grad_output[0].detach()]
        if self.debug:
            print(f"Gradient {name} shape: {grad_output[0].shape}")
            print(f"Gradient {name} range: [{grad_output[0].min().item():.2f}, {grad_output[0].max().item():.2f}]")
            
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
        
    def _compute_gradcam(self, layer_name, target_size, kidney_mask=None):
        """Compute Grad-CAM attention map for a specific layer"""
        acts = self.activations[layer_name]
        grads = self.gradients[layer_name]
        
        if acts is None or grads is None or len(acts) == 0 or len(grads) == 0:
            if self.debug:
                print(f"Missing activations or gradients for layer {layer_name}")
            return None
            
        acts, grads = acts[0], grads[0]
        
        if self.debug:
            print(f"\nComputing GradCAM for layer {layer_name}:")
            print(f"Activations shape: {acts.shape}")
            print(f"Activations range: [{acts.min().item():.2f}, {acts.max().item():.2f}]")
            print(f"Gradients shape: {grads.shape}")
            print(f"Gradients range: [{grads.min().item():.2f}, {grads.max().item():.2f}]")
        
        try:
            # Keep batch dimension for proper broadcasting
            weights = grads.mean(dim=(2, 3, 4), keepdim=True)  # Shape: (1, C, 1, 1, 1)
            
            if self.debug:
                print(f"Weights shape: {weights.shape}")
                print(f"Weights range: [{weights.min().item():.2f}, {weights.max().item():.2f}]")
            
            # Check for zero weights
            if (weights == 0).all():
                if self.debug:
                    print(f"Warning: All weights are zero for layer {layer_name}")
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
                print(f"Warning: NaN values in CAM for layer {layer_name}")
                return None
            
            # Normalize
            cam_min = cam.min()
            cam_max = cam.max()
            if cam_min == cam_max:
                if self.debug:
                    print(f"Warning: Constant CAM values for layer {layer_name}")
                return None
            
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
            
            # Upsample to target size
            cam = F.interpolate(
                cam.unsqueeze(0),  # Add channel dim for interpolate
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            
            # Apply kidney mask if provided
            if kidney_mask is not None:
                if self.debug:
                    print("Applying kidney mask...")
                # Convert mask to tensor and move to device
                mask = torch.from_numpy(kidney_mask).float().to(cam.device)
                # Expand dimensions to match cam
                mask = mask.unsqueeze(0).unsqueeze(0)
                # Apply mask
                cam = cam * mask
                # Renormalize
                cam_min = cam.min()
                cam_max = cam.max()
                if cam_min != cam_max:
                    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
            
            if self.debug:
                print(f"Final CAM shape: {cam.shape}")
                print(f"Final CAM range: [{cam.min().item():.2f}, {cam.max().item():.2f}]")
            
            return cam.squeeze().cpu().numpy()
            
        except Exception as e:
            if self.debug:
                print(f"Error in GradCAM computation for layer {layer_name}: {str(e)}")
                import traceback
                print(traceback.format_exc())
            return None
            
    def _combine_attention_maps(self, attention_maps, weights=None):
        """Combine multiple attention maps with optional weighting"""
        if not attention_maps:
            return None
            
        if weights is None:
            weights = {name: 1.0 for name in attention_maps.keys()}
            
        # Combine maps
        combined = None
        total_weight = 0
        for name, attention in attention_maps.items():
            if attention is not None:
                weight = weights.get(name, 1.0)
                if combined is None:
                    combined = weight * attention
                else:
                    combined += weight * attention
                total_weight += weight
                
        if combined is None:
            return None
            
        # Normalize
        if total_weight > 0:
            combined /= total_weight
            
        return combined
        
    def _compute_predictions(self, output, window_shape):
        """Compute prediction probabilities"""
        # Apply softmax first
        probs = F.softmax(output, dim=1)
        
        # Get output shape
        batch, channels, d, h, w = probs.shape
        
        # Get target shape
        target_d, target_h, target_w = window_shape
        
        if self.debug:
            print(f"Prediction shapes - Input: {(d,h,w)}, Target: {(target_d,target_h,target_w)}")
            
        # Interpolate each channel independently to preserve dimensionality
        pred_full = []
        for c in range(channels):
            # Add extra dimensions for batch and channel
            channel_data = probs[:,c:c+1]
            
            # Upsample to target size
            upsampled = F.interpolate(
                channel_data,
                size=(target_d, target_h, target_w),
                mode='trilinear',
                align_corners=False
            )
            pred_full.append(upsampled)
            
        # Stack channels back together
        pred_full = torch.cat(pred_full, dim=1)
        
        return pred_full[0].cpu().numpy()

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
        
    def sliding_window_inference(self, volume, kidney_mask=None):
        """Perform sliding window inference with Grad-CAM"""
        print("\nStarting sliding window inference with Grad-CAM...")
        
        # Layer weights for attention map combination
        layer_weights = {
            'encoder_0': 0.2,
            'encoder_1': 0.3,
            'encoder_2': 0.5,
            'encoder_3': 0.8,
            'bottleneck': 1.0
        }
        
        # Preprocessing
        volume = self.preprocess_volume(volume)
        padded_volume, padding = self._pad_volume(volume)
        d, h, w = padded_volume.shape
        stride = self._compute_stride()
        
        # Initialize output arrays
        predictions = np.zeros((2,) + padded_volume.shape, dtype=np.float32)
        attention_maps = np.zeros_like(padded_volume)
        count_map = np.zeros_like(padded_volume)
        
        # Pad kidney mask if provided
        if kidney_mask is not None:
            padded_mask, _ = self._pad_volume(kidney_mask)
        else:
            padded_mask = None
        
        # Sliding window inference
        total_windows = ((d-1)//stride[0] + 1) * ((h-1)//stride[1] + 1) * ((w-1)//stride[2] + 1)
        with tqdm(total=total_windows) as pbar:
            for z in range(0, d - self.window_size[0] + 1, stride[0]):
                for y in range(0, h - self.window_size[1] + 1, stride[1]):
                    for x in range(0, w - self.window_size[2] + 1, stride[2]):
                        try:
                            # Extract window
                            window = padded_volume[
                                z:z + self.window_size[0],
                                y:y + self.window_size[1],
                                x:x + self.window_size[2]
                            ]
                            
                            # Extract corresponding mask window if available
                            mask_window = None
                            if padded_mask is not None:
                                mask_window = padded_mask[
                                    z:z + self.window_size[0],
                                    y:y + self.window_size[1],
                                    x:x + self.window_size[2]
                                ]
                            
                            window_tensor = self._preprocess_window(window)
                            
                            if self.debug:
                                print(f"\nProcessing window at ({z},{y},{x})")
                                print(f"Window shape: {window.shape}")
                                print(f"Window tensor shape: {window_tensor.shape}")
                            
                            # Forward pass
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
                                             
                            # Compute Grad-CAM for each layer
                            self.model.zero_grad()
                            target_score = output[0, 1].max()
                            
                            # Only compute GradCAM if prediction score is significant
                            if target_score > 0.5:
                                # Compute gradients
                                target_score.backward(retain_graph=True)
                                
                                # Get attention maps from each layer
                                layer_attention_maps = {}
                                for layer_name in self.target_layers.keys():
                                    attention = self._compute_gradcam(
                                        layer_name,
                                        self.window_size,
                                        mask_window
                                    )
                                    if attention is not None:
                                        layer_attention_maps[layer_name] = attention
                                
                                # Combine attention maps
                                combined_attention = self._combine_attention_maps(
                                    layer_attention_maps,
                                    layer_weights
                                )
                                
                                if combined_attention is not None:
                                    attention_maps[z:z + self.window_size[0],
                                                y:y + self.window_size[1],
                                                x:x + self.window_size[2]] += combined_attention
                                    count_map[z:z + self.window_size[0],
                                            y:y + self.window_size[1],
                                            x:x + self.window_size[2]] += 1
                                            
                            # Clear cache
                            self.activations = {name: [] for name in self.target_layers.keys()}
                            self.gradients = {name: [] for name in self.target_layers.keys()}
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

    def process_case(self, case_path):
        """Process a full case from the dataset"""
        # Load image
        img_path = case_path / "imaging.nii.gz"
        if not img_path.exists():
            img_path = case_path / "raw_data" / "imaging.nii.gz"
            
        img_obj = nib.load(str(img_path))
        volume = img_obj.get_fdata()
        
        # Try to load kidney segmentation if available
        try:
            kidney_mask_path = case_path / "kidney_segmentation.nii.gz"
            if kidney_mask_path.exists():
                kidney_mask = nib.load(str(kidney_mask_path)).get_fdata()
                kidney_mask = (kidney_mask > 0).astype(np.float32)
            else:
                kidney_mask = None
        except:
            kidney_mask = None
            if self.debug:
                print("Could not load kidney segmentation mask")
        
        predictions, attention_maps = self.sliding_window_inference(volume, kidney_mask)
        
        return volume, predictions, attention_maps
