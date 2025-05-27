import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_convs=2):
        super().__init__()
        ops = []
        for i in range(n_convs):
            channels = in_channels if i == 0 else out_channels
            ops.append(ConvBlock(channels, out_channels))
        self.conv_block = nn.Sequential(*ops)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        features = self.conv_block(x)
        return self.pool(features), features

class DetectionModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, features=(32, 64, 128, 256, 320)):
        super().__init__()
        
        # Encoder
        self.down_blocks = nn.ModuleList()
        current_channels = in_channels
        for feature in features:
            self.down_blocks.append(DownBlock(current_channels, feature))
            current_channels = feature

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(features[-1], features[-1] // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(features[-1] // 2, num_classes)
        )
        
        # For Grad-CAM
        self.target_layer = None
        self.gradients = None
        self.activations = None
    
    def _get_activation(self, module, input, output):
        self.activations = output
    
    def _get_gradient(self, module, input_grad, output_grad):
        self.gradients = output_grad[0]
    
    def set_target_layer(self, layer_name):
        if hasattr(self, layer_name):
            target_layer = getattr(self, layer_name)
            target_layer.register_forward_hook(self._get_activation)
            target_layer.register_full_backward_hook(self._get_gradient)
            self.target_layer = target_layer

    def forward(self, x):
        # Encoder pathway with skip connections
        for down_block in self.down_blocks:
            x, _ = down_block(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        return x