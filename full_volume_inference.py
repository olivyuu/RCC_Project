import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import nibabel as nib
from tqdm import tqdm

class FullVolumeInference:
    def __init__(self, model, config, debug=False):
        """Initialize full volume inference handler"""
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
        
        # Look for encoder[3] first, then bottleneck as fallback
        if hasattr(self.model, 'encoder') and len(self.model.encoder) > 3:
            target_layer = self.model.encoder[3]
        else:
            # Search for bottleneck
            for name, module in self.model.named_modules():
                if 'bottleneck' in name.lower():
                    target_layer = module
                    break
        
        if target_layer is None:
            raise ValueError("Could not find suitable target layer for Grad-CAM")
            
        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)
        
        if self.debug:
            print(f"Registered Grad-CAM hooks on layer: {target_layer}")
            
        return target_layer
        
    def _save_activation(self, module, input, output):
        """Hook to save layer activations"""
        self.activations = [output.detach()]
        
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save layer gradients"""
        self.gradients = [grad_output[0].detach()]
        
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
        
    def _forward_pass(self, window_tensor):
        """Perform forward pass through the model"""
        window_tensor.requires_grad = True
        output = self.model(window_tensor)
        if isinstance(output, list):
            output = output[0]
        return output
        
    def _compute_predictions(self, output, window_shape):
        """Compute prediction probabilities"""
        pred_full = F.interpolate(
            F.softmax(output, dim=1),
            size=window_shape,
            mode='trilinear',
            align_corners=False
        )
        return pred_full[0].cpu().numpy()
        
    def _compute_gradcam(self, acts, grads, target_size):
        """Compute Grad-CAM attention map"""
        if acts is None or grads is None or len(acts) == 0 or len(grads) == 0:
            return None
            
        acts, grads = acts[0], grads[0]
        
        if acts.shape[1] != grads.shape[1]:
            if self.debug:
                print(f"Channel dimension mismatch - acts: {acts.shape[1]}, grads: {grads.shape[1]}")
            return None
            
        # Compute channel-wise weights and weighted sum
        weights = grads.mean(dim=(2, 3, 4)).view(-1, acts.shape[1], 1, 1, 1)
        cam = (weights * acts).sum(dim=1)
        cam = F.relu(cam)
        
        if torch.isnan(cam).any():
            return None
            
        # Normalize and resize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam_full = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
        
        return cam_full.squeeze().cpu().numpy()
        
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
                            
                            # Forward pass and predictions
                            output = self._forward_pass(window_tensor)
                            
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
                            output[0, 1].max().backward(retain_graph=True)
                            
                            # Generate attention map
                            attention = self._compute_gradcam(
                                self.activations,
                                self.gradients,
                                window_tensor.shape[2:]
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
        img_path = case_path / "imaging.nii.gz"
        if not img_path.exists():
            img_path = case_path / "raw_data" / "imaging.nii.gz"
            
        img_obj = nib.load(str(img_path))
        volume = img_obj.get_fdata()
        
        predictions, attention_maps = self.sliding_window_inference(volume)
        
        return volume, predictions, attention_maps
