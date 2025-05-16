import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import nibabel as nib
from tqdm import tqdm

class FullVolumeInference:
    def __init__(self, model, config, debug=False):
        """
        Initialize full volume inference handler
        
        Args:
            model: The trained model
            config: Configuration object
        """
        self.model = model
        self.config = config
        self.debug = debug  # Store debug flag
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.window_size = (64, 128, 128)  # Size that our volume model expects
        self.overlap = 0.25  # 25% overlap for full volumes
        
        # Initialize Grad-CAM storage as lists
        self.activations = []
        self.gradients = []
        
        # Set up Grad-CAM hooks
        def save_gradient(module, grad_input, grad_output):
            self.gradients = [grad_output[0].detach()]  # Store as single-item list
            
        def save_output(module, input, output):
            self.activations = [output.detach()]  # Store as single-item list
            
        # Find and register hooks for bottleneck layer
        for name, module in self.model.named_modules():
            if 'bottleneck' in name:
                module.register_forward_hook(save_output)
                module.register_full_backward_hook(save_gradient)
                if self.debug:
                    print(f"Registered Grad-CAM hooks on {name}")
                break
        
    def _compute_stride(self):
        """Compute stride based on window size and overlap"""
        return [int(ws * (1 - self.overlap)) for ws in self.window_size]
        
    def preprocess_volume(self, volume):
        """
        Preprocess the input volume
        
        Args:
            volume: Input numpy array [D, H, W]
            
        Returns:
            Preprocessed volume
        """
        # Convert to float32
        volume = volume.astype(np.float32)
        
        # Clip to training data range
        volume = np.clip(volume, -1024, 3071)
        
        # Normalize to [0, 1]
        volume = (volume - (-1024)) / (3071 - (-1024))
        
        # Scale to [-1, 1]
        volume = volume * 2 - 1
            
        return volume
        
    def _pad_volume(self, volume):
        """
        Pad volume to handle window extraction
        """
        # Get original size
        d, h, w = volume.shape
        
        # Calculate required padding
        pad_d = (self.window_size[0] - d % self.window_size[0]) % self.window_size[0]
        pad_h = (self.window_size[1] - h % self.window_size[1]) % self.window_size[1]
        pad_w = (self.window_size[2] - w % self.window_size[2]) % self.window_size[2]
        
        # Pad the volume
        padded = np.pad(volume,
                       ((0, pad_d), (0, pad_h), (0, pad_w)),
                       mode='constant')
        
        return padded, (pad_d, pad_h, pad_w)
        
    def sliding_window_inference(self, volume):
        """
        Perform sliding window inference on the full volume with Grad-CAM
        
        Args:
            volume: Input numpy array [D, H, W]
            
        Returns:
            predictions: Full volume predictions
            attention_maps: Full volume attention maps
        """
        print("\nStarting sliding window inference with Grad-CAM...")
        # Preprocess volume using training pipeline
        volume = self.preprocess_volume(volume)
        
        # Pad if needed
        padded_volume, padding = self._pad_volume(volume)
        d, h, w = padded_volume.shape
        
        # Calculate stride
        stride = self._compute_stride()
        
        # Initialize output tensors with channels first for predictions
        predictions = np.zeros((2,) + padded_volume.shape, dtype=np.float32)  # [C, D, H, W]
        attention_maps = np.zeros_like(padded_volume)
        count_map = np.zeros_like(padded_volume)  # Track overlapping regions
        
        # No need to register hooks again - already done in __init__
        
        # Sliding window inference with progress bar
        with tqdm(total=(d//stride[0] + 1) * (h//stride[1] + 1) * (w//stride[2] + 1)) as pbar:
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
                            
                            # Prepare input
                            window_tensor = torch.from_numpy(window).float()
                            window_tensor = window_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
                            window_tensor = window_tensor.to(self.device)
                            
                            # Forward pass with gradient computation
                            window_tensor.requires_grad = True
                            output = self.model(window_tensor)
                            if isinstance(output, list):
                                output = output[0]
                                
                            # Get raw predictions first
                            with torch.no_grad():
                                probs = F.softmax(output, dim=1)[0, 1]  # [32, 64, 64]
                                max_prob = probs.max().item()
                                
                                if self.debug and z == 0 and y == 0 and x == 0:
                                    print(f"\nWindow at ({z},{y},{x}):")
                                    print(f"Output shape: {output.shape}")
                                    print(f"Max probability: {max_prob:.3f}")
                                
                                # Skip low probability regions
                                if max_prob < 0.5:
                                    continue
                                    
                                # Store predictions
                                pred_full = F.interpolate(
                                    F.softmax(output, dim=1),
                                    size=window_tensor.shape[2:],
                                    mode='trilinear',
                                    align_corners=False
                                )
                                predictions[:, z:z + self.window_size[0],
                                         y:y + self.window_size[1],
                                         x:x + self.window_size[2]] += pred_full[0].cpu().numpy()

                            # Compute Grad-CAM
                            self.model.zero_grad()
                            output[0, 1].max().backward(retain_graph=True)
                            
                            if self.activations is None or self.gradients is None:
                                if self.debug:
                                    print("Warning: Missing activations or gradients")
                                continue
                            
                            # Debug info for hooks
                            if self.debug and z == 0 and y == 0 and x == 0 and len(self.activations) > 0 and len(self.gradients) > 0:
                                print(f"Activations shape: {self.activations[0].shape}")
                                print(f"Activations range: [{self.activations[0].min().item():.2f}, {self.activations[0].max().item():.2f}]")
                                print(f"Gradients shape: {self.gradients[0].shape}")
                                print(f"Gradients range: [{self.gradients[0].min().item():.2f}, {self.gradients[0].max().item():.2f}]")
                            
                            # Compute CAM from hooks
                            if len(self.activations) > 0 and len(self.gradients) > 0:
                                acts = self.activations[0]  # Get stored activation
                                grads = self.gradients[0]   # Get stored gradient
                                
                                # Get activation and gradient weights
                                weights = grads.mean(dim=(1, 2, 3))  # Average over spatial dims
                                cam = (weights.view(-1, 1, 1, 1) * acts).sum(0)  # Weighted sum
                                cam = F.relu(cam)  # Keep only positive contributions
                                
                                if not torch.isnan(cam).any():
                                    # Normalize and interpolate
                                    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                                    cam_full = F.interpolate(
                                        cam.unsqueeze(0).unsqueeze(0),
                                        size=window_tensor.shape[2:],
                                        mode='trilinear',
                                        align_corners=False
                                    ).squeeze()
                                    
                                    # Store attention map
                                    attention_maps[z:z + self.window_size[0],
                                                y:y + self.window_size[1],
                                                x:x + self.window_size[2]] += cam_full.cpu().numpy()
                                    count_map[z:z + self.window_size[0],
                                            y:y + self.window_size[1],
                                            x:x + self.window_size[2]] += 1
                                    
                                    del cam_full
                                del cam
                            
                            # Clear hook storage
                            self.activations = []  # Reset to empty list
                            self.gradients = []    # Reset to empty list
                            
                            # Clear other tensors
                            del window_tensor, output
                            torch.cuda.empty_cache()
                            
                        except Exception as e:
                            print(f"Error processing window at ({z},{y},{x}): {str(e)}")
                            continue
                            
                        pbar.update(1)
                                
        # Average overlapping regions
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
        """
        Process a full case from the dataset
        
        Args:
            case_path: Path to the case directory
            
        Returns:
            Original volume, predictions, and attention maps
        """
        # Load the image
        img_path = case_path / "imaging.nii.gz"
        if not img_path.exists():
            img_path = case_path / "raw_data" / "imaging.nii.gz"
            
        img_obj = nib.load(str(img_path))
        volume = img_obj.get_fdata()
        
        # Run inference
        predictions, attention_maps = self.sliding_window_inference(volume)
        
        return volume, predictions, attention_maps
