import torch
import numpy as np
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from typing import List, Callable

def reshape_3d_transform(tensor):
    """
    Reshape function for 3D volumes
    Args:
        tensor: Input tensor of shape (batch_size, channels, D, H, W)
    Returns:
        Reshaped tensor suitable for GradCAM
    """
    # Check input
    if tensor.dim() != 5:
        raise ValueError(f"Expected 5D tensor (B,C,D,H,W), got shape {tensor.shape}")
    
    # For 3D, we want to keep spatial dimensions together
    # (B, C, D, H, W) -> (B, C, H, W, D)
    tensor = tensor.permute(0, 1, 3, 4, 2)
    return tensor

class GradCAMPlusPlus3D(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=True):
        super().__init__(model, target_layers, reshape_transform=reshape_3d_transform)
        self.cuda = use_cuda

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
        alpha = grad_2 / (2 * grad_2 + (grad_3 * activations).sum(-1, keepdim=True) + 1e-7)
        weights = (alpha * torch.relu(grads)).sum(-1)
        
        return weights.cpu().numpy()

    def forward(self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool = False) -> np.ndarray:
        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                 requires_grad=True)

        outputs = self.activations_and_grads(input_tensor)
        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output) for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                  targets,
                                                  eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)

class MultiScaleGradCAM:
    def __init__(self, model, target_layers=None, weights=None, use_cuda=True):
        """
        Initialize MultiScaleGradCAM with multiple target layers
        
        Args:
            model: The model to analyze
            target_layers: List of layers to generate CAM from
            weights: Optional list of weights for each layer's contribution
            use_cuda: Whether to use GPU
        """
        self.model = model
        self.use_cuda = use_cuda
        
        # Get target layers if not provided
        if target_layers is None:
            target_layers = self.get_target_layers_from_model(model)
        self.target_layers = target_layers
        
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
        if self.use_cuda:
            input_tensor = input_tensor.cuda()
            
        # Get CAM from each layer
        attention_maps = []
        for extractor, weight in zip(self.cam_extractors, self.weights):
            try:
                # Generate CAM
                targets = [ClassifierOutputTarget(target_category)]
                cam = extractor(input_tensor=input_tensor,
                              targets=targets,
                              eigen_smooth=False)  # (N, D, H, W)
                
                # Move depth dimension back to middle
                cam = np.moveaxis(cam, 1, -1)  # (N, H, W, D)
                
                # Normalize and weight
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                cam = cam * weight
                attention_maps.append(cam)
            except Exception as e:
                print(f"Error generating CAM for layer: {str(e)}")
                continue
            
        if not attention_maps:
            raise ValueError("No valid attention maps generated")
            
        # Combine attention maps
        combined_cam = np.sum(attention_maps, axis=0)
        combined_cam = np.clip(combined_cam, 0, 1)
        
        # Move depth back to first dimension for visualization
        combined_cam = np.moveaxis(combined_cam, -1, 1)[0]  # (D, H, W)
        return combined_cam
        
    @staticmethod
    def get_target_layers_from_model(model):
        """Helper to get common target layers from nnUNetv2"""
        # Find the encoder blocks (last conv layer in each block)
        encoder_layers = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv3d):
                encoder_layers.append(module)
                
        if len(encoder_layers) == 0:
            raise ValueError("No Conv3d layers found in model")
            
        # Take the last encoder layer, middle layer, and first decoder layer
        if len(encoder_layers) >= 3:
            target_layers = [
                encoder_layers[-1],  # Deep encoder features
                encoder_layers[len(encoder_layers)//2],  # Middle features
                encoder_layers[0]  # Early features
            ]
        else:
            # If fewer layers, just use what we have
            target_layers = encoder_layers
        
        return target_layers
        
    @staticmethod
    def get_default_layer_weights():
        """Get default weights for combining attention maps"""
        return [0.5, 0.3, 0.2]  # Emphasize deep features more