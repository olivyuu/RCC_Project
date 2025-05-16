import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import nibabel as nib
from tqdm import tqdm

class FullVolumeInference:
    def __init__(self, model, config):
        """
        Initialize full volume inference handler
        
        Args:
            model: The trained model
            config: Configuration object
        """
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.window_size = (64, 128, 128)  # Size that our volume model expects
        self.overlap = 0.25  # 25% overlap for full volumes
        
    def _compute_stride(self):
        """Compute stride based on window size and overlap"""
        return [int(ws * (1 - self.overlap)) for ws in self.window_size]
        
    def preprocess_volume(self, volume):
        """
        Preprocess the input volume using same preprocessing as training
        
        Args:
            volume: Input numpy array [D, H, W]
            
        Returns:
            Preprocessed volume
        """
        from preprocessing.preprocessor import KiTS23Preprocessor
        
        # Initialize preprocessor with same config
        preprocessor = KiTS23Preprocessor(self.config)
        
        # Preprocess volume using training pipeline
        volume = preprocessor.preprocess_volume(volume)
            
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
        
        # Enable gradient computation for Grad-CAM
        if not hasattr(self.model, 'target_layers'):
            print("Setting up Grad-CAM target layers...")
            # For UNet, we typically want the bottleneck layer
            for name, module in self.model.named_modules():
                if 'bottleneck' in name:
                    self.model.target_layers = [module]
                    break
        
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
                                
                            # Generate Grad-CAM for tumor class
                            tumor_class_score = output[0, 1].sum()  # Sum predictions for tumor class
                            self.model.zero_grad()
                            tumor_class_score.backward()
                            
                            # Get activations from target layer
                            target_layer = self.model.target_layers[0]
                            activations = target_layer.activations
                            gradients = target_layer.gradients
                            
                            # Calculate attention weights
                            weights = torch.mean(gradients, dim=(2, 3, 4), keepdim=True)
                            cam = torch.sum(weights * activations, dim=1)
                            
                            # Process CAM
                            cam = F.relu(cam)  # Apply ReLU to focus on positive contributions
                            cam = F.interpolate(
                                cam.unsqueeze(1),  # [B, 1, D, H, W]
                                size=window_tensor.shape[2:],  # Original window size
                                mode='trilinear',
                                align_corners=False
                            ).squeeze()
                        
                            # Normalize CAM
                            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                            
                            # Add predictions and attention to output tensors
                            with torch.no_grad():
                                output = F.softmax(output, dim=1)
                                predictions[:, z:z + self.window_size[0],
                                            y:y + self.window_size[1],
                                            x:x + self.window_size[2]] += output[0].cpu().numpy()
                                
                                attention_maps[z:z + self.window_size[0],
                                            y:y + self.window_size[1],
                                            x:x + self.window_size[2]] += cam.cpu().numpy()
                                
                                count_map[z:z + self.window_size[0],
                                        y:y + self.window_size[1],
                                        x:x + self.window_size[2]] += 1
                                
                            # Clear GPU memory
                            del window_tensor, output, cam
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