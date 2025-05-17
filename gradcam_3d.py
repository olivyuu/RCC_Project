import torch
import numpy as np
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.base_cam import BaseCAM
from typing import List, Callable

class GradCAMPlusPlus3D(GradCAMPlusPlus):
    def __init__(self, model, target_layers, use_cuda=True,
                 reshape_transform=None):
        super().__init__(model, target_layers, use_cuda, reshape_transform)
        
    def get_cam_weights(self,
                       input_tensor: torch.Tensor,
                       target_layer: torch.nn.Module,
                       target_category: int,
                       activations: torch.Tensor,
                       grads: torch.Tensor) -> np.ndarray:
        # Adapt GradCAM++ for 3D volumes
        grads = grads.reshape(grads.shape[0], -1)  # Flatten spatial dimensions
        activations = activations.reshape(activations.shape[0], -1)
        
        grad_2 = grads.pow(2)
        grad_3 = grads.pow(3)
        alpha = grad_2 / (2 * grad_2 + (grad_3 * activations).sum(-1, keepdim=True))
        weights = (alpha * torch.relu(grads)).sum(-1)
        
        return weights.cpu().numpy()

class MultiScaleGradCAM:
    def __init__(self, model, target_layers, weights=None, use_cuda=True):
        """
        Initialize MultiScaleGradCAM with multiple target layers
        
        Args:
            model: The model to analyze
            target_layers: List of layers to generate CAM from
            weights: Optional list of weights for each layer's contribution
            use_cuda: Whether to use GPU
        """
        self.model = model
        self.target_layers = target_layers
        self.use_cuda = use_cuda
        
        # Default to equal weights if none provided
        if weights is None:
            weights = [1.0/len(target_layers)] * len(target_layers)
        self.weights = weights
        
        # Create GradCAM++ for each target layer
        self.cam_extractors = [
            GradCAMPlusPlus3D(
                model=model,
                target_layers=[layer],
                use_cuda=use_cuda
            )
            for layer in target_layers
        ]
        
    def __call__(self, input_tensor: torch.Tensor, target_category: int) -> np.ndarray:
        """
        Generate combined multi-scale GradCAM++
        
        Args:
            input_tensor: Input to analyze (N, C, D, H, W)
            target_category: Category to generate CAM for
            
        Returns:
            Combined attention map (D, H, W)
        """
        # Get CAM from each layer
        attention_maps = []
        for extractor, weight in zip(self.cam_extractors, self.weights):
            # Generate CAM
            cam = extractor(input_tensor=input_tensor,
                          target_category=target_category,
                          eigen_smooth=False)  # (N, D, H, W)
            
            # Normalize and weight
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cam = cam * weight
            attention_maps.append(cam)
            
        # Combine attention maps
        combined_cam = np.sum(attention_maps, axis=0)
        combined_cam = np.clip(combined_cam, 0, 1)
        
        return combined_cam[0]  # Remove batch dimension
        
    @staticmethod
    def get_target_layers_from_model(model):
        """Helper to get common target layers from nnUNetv2"""
        target_layers = [
            model.model.encoder[-1],  # Deep encoder features
            model.model.bottleneck,   # Bottleneck features  
            model.model.decoder[0]    # High-level decoder features
        ]
        return target_layers
        
    @staticmethod
    def get_default_layer_weights():
        """Get default weights for combining attention maps"""
        return [0.5, 0.3, 0.2]  # Emphasize deep features more