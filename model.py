import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvDropoutNormNonlin(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 conv_op=nn.Conv3d, norm_op=nn.InstanceNorm3d):
        super().__init__()
        self.conv = conv_op(in_channels, out_channels, 3, padding=1)
        self.norm = norm_op(out_channels)
        self.nonlin = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.nonlin(x)
        x = self.dropout(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_convs=2):
        super().__init__()
        ops = []
        for i in range(n_convs):
            channels = in_channels if i == 0 else out_channels
            ops.append(ConvDropoutNormNonlin(channels, out_channels))
        self.conv_block = nn.Sequential(*ops)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        skip = self.conv_block(x)
        return self.pool(skip), skip

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, 
                                        kernel_size=2, stride=2)
        # After concatenation, the number of channels is doubled
        self.conv_block = nn.Sequential(
            ConvDropoutNormNonlin(in_channels + out_channels, out_channels),
            ConvDropoutNormNonlin(out_channels, out_channels)
        )

    def forward(self, x, skip):
        x = self.upconv(x)
        # Handle different sized feature maps
        if x.shape != skip.shape:
            x = F.interpolate(x, skip.shape[2:])
        x = torch.cat((skip, x), dim=1)
        return self.conv_block(x)

class nnUNetv2(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, 
                 features=(32, 64, 128, 256, 320)):
        super().__init__()
        
        # Encoder
        self.down_blocks = nn.ModuleList()
        current_channels = in_channels
        for feature in features:
            self.down_blocks.append(DownBlock(current_channels, feature))
            current_channels = feature

        # Bottleneck
        self.bottleneck = ConvDropoutNormNonlin(features[-1], features[-1])

        # Decoder with deep supervision
        self.up_blocks = nn.ModuleList()
        self.deep_supervision = nn.ModuleList()
        
        features = list(reversed(features))
        for i in range(len(features) - 1):
            self.up_blocks.append(UpBlock(features[i], features[i + 1]))
            self.deep_supervision.append(
                nn.Conv3d(features[i + 1], out_channels, 1)
            )

        self.final_conv = nn.Conv3d(features[-1], out_channels, 1)
        self.target_layer = None
        self.gradients = None
        self.activations = None
    
    def _get_activation(self, module, input, output):
        self.activations = output
    
    def _get_gradient(self, module, input_grad, output_grad):
        self.gradients = output_grad[0]
    
    def set_target_layer(self, layer_name):
        """Set the target layer for Grad-CAM visualization"""
        if hasattr(self, layer_name):
            target_layer = getattr(self, layer_name)
            target_layer.register_forward_hook(self._get_activation)
            target_layer.register_full_backward_hook(self._get_gradient)
            self.target_layer = target_layer

    def forward(self, x):
        # Encoder
        skip_connections = []
        for down_block in self.down_blocks[:-1]:
            x, skip = down_block(x)
            skip_connections.append(skip)
        
        x, skip = self.down_blocks[-1](x)
        x = self.bottleneck(x)
        skip_connections.append(skip)
        skip_connections = skip_connections[::-1]  # Reverse for decoder

        # Decoder with deep supervision
        deep_outputs = []
        for i, up_block in enumerate(self.up_blocks):
            x = up_block(x, skip_connections[i])
            if self.training:
                deep_outputs.append(self.deep_supervision[i](x))

        output = self.final_conv(x)

        if self.training and len(deep_outputs) > 0:
            # Return main output and deep supervision outputs during training
            return [output] + deep_outputs
        return output