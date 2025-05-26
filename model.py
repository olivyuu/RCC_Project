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

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = self.sigmoid(avg_out + max_out).view(x.size(0), x.size(1), 1, 1, 1)
        return x * out

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_convs=2):
        super().__init__()
        ops = []
        for i in range(n_convs):
            channels = in_channels if i == 0 else out_channels
            ops.append(ConvDropoutNormNonlin(channels, out_channels))
        self.conv_block = nn.Sequential(*ops)
        self.pool = nn.MaxPool3d(2)
        self.channel_attention = ChannelAttention(out_channels)
        self.spatial_attention = SpatialAttention(out_channels)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return self.pool(x), x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        print(f"Creating UpBlock with in_channels={in_channels}, out_channels={out_channels}")
        
        # Keep original upconv behavior that matches checkpoint
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, 
                                      kernel_size=2, stride=2)
        
        # Reduce skip connection channels to match upconv output
        self.reduce_skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        
        # Initialize attention gate with correct dimensions
        self.attention_gate = AttentionGate(F_g=out_channels, F_l=out_channels, F_int=out_channels//2)
        
        # Calculate concat channels based on checkpoint dimensions
        if in_channels == 320:
            concat_channels = 576  # From checkpoint
        elif in_channels == 256:
            concat_channels = 384  # From checkpoint
        elif in_channels == 128:
            concat_channels = 192  # From checkpoint
        elif in_channels == 64:
            concat_channels = 96   # From checkpoint
        else:
            concat_channels = out_channels * 2  # Fallback
            
        print(f"UpBlock concat_channels={concat_channels}")
        
        self.conv_block = nn.Sequential(
            ConvDropoutNormNonlin(concat_channels, out_channels),
            ConvDropoutNormNonlin(out_channels, out_channels)
        )
        self.channel_attention = ChannelAttention(out_channels)
        self.spatial_attention = SpatialAttention(out_channels)

    def forward(self, x, skip):
        print(f"\nUpBlock forward:")
        print(f"Input shape: {x.shape}")
        print(f"Skip shape: {skip.shape}")
        
        x = self.upconv(x)
        print(f"After upconv shape: {x.shape}")
        
        # Handle different sized feature maps
        if x.shape != skip.shape:
            x = F.interpolate(x, skip.shape[2:])
            print(f"After interpolate shape: {x.shape}")
        
        # Reduce skip connection channels
        skip = self.reduce_skip(skip)
        print(f"After reducing skip channels: {skip.shape}")
        
        # Apply attention gate
        skip = self.attention_gate(x, skip)
        print(f"After attention gate shape: {skip.shape}")
        
        # Concatenate skip connection
        x = torch.cat((skip, x), dim=1)
        print(f"After concatenation shape: {x.shape}")
        
        x = self.conv_block(x)
        print(f"After conv block shape: {x.shape}")
        
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        print(f"Creating AttentionGate with F_g={F_g}, F_l={F_l}, F_int={F_int}")
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class DeepSupervisionHead(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv(x)
        if self.scale_factor > 1:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='trilinear', align_corners=False)
        return x

class nnUNetv2(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, 
                 features=(32, 64, 128, 256, 320)):  # Original dimensions
        super().__init__()
        print(f"\nInitializing nnUNetv2:")
        print(f"Input channels: {in_channels}")
        print(f"Output channels: {out_channels}")
        print(f"Feature dimensions: {features}")
        
        # Encoder
        self.down_blocks = nn.ModuleList()
        current_channels = in_channels
        for feature in features:
            print(f"Adding DownBlock: {current_channels} -> {feature}")
            self.down_blocks.append(DownBlock(current_channels, feature))
            current_channels = feature

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvDropoutNormNonlin(features[-1], features[-1]),
            ChannelAttention(features[-1]),
            SpatialAttention(features[-1])
        )

        # Decoder with deep supervision and attention
        self.up_blocks = nn.ModuleList()
        self.deep_supervision = nn.ModuleList()
        
        features = list(reversed(features))
        scales = [8, 4, 2, 1]  # Scale factors for deep supervision
        for i in range(len(features) - 1):
            print(f"Adding UpBlock {i}: {features[i]} -> {features[i + 1]}")
            self.up_blocks.append(UpBlock(features[i], features[i + 1]))
            self.deep_supervision.append(
                DeepSupervisionHead(features[i + 1], out_channels, scales[i])
            )

        self.final_conv = nn.Conv3d(features[-1], out_channels, 1)
        
        # Progressive learning weights
        self.progressive_weights = nn.Parameter(torch.ones(len(self.deep_supervision) + 1))
        self.softmax = nn.Softmax(dim=0)
        
        # For Grad-CAM
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
        print(f"\nnnUNetv2 forward pass:")
        print(f"Input shape: {x.shape}")
        
        # Get normalized progressive weights
        if self.training:
            weights = self.softmax(self.progressive_weights)
        
        # Encoder
        skip_connections = []
        for i, down_block in enumerate(self.down_blocks[:-1]):
            x, skip = down_block(x)
            skip_connections.append(skip)
            print(f"Down block {i} output shape: {x.shape}")
            print(f"Skip connection {i} shape: {skip.shape}")
        
        x, skip = self.down_blocks[-1](x)
        print(f"Final down block output shape: {x.shape}")
        print(f"Final skip connection shape: {skip.shape}")
        
        x = self.bottleneck(x)
        print(f"After bottleneck shape: {x.shape}")
        
        skip_connections.append(skip)
        skip_connections = skip_connections[::-1]  # Reverse for decoder

        # Decoder with deep supervision
        deep_outputs = []
        for i, up_block in enumerate(self.up_blocks):
            print(f"\nUp block {i}:")
            x = up_block(x, skip_connections[i])
            print(f"Up block {i} output shape: {x.shape}")
            if self.training:
                deep_out = self.deep_supervision[i](x)
                deep_outputs.append(deep_out)

        output = self.final_conv(x)
        print(f"Final output shape: {output.shape}")

        if self.training:
            # Weight outputs according to progressive learning weights
            outputs = [output] + deep_outputs
            weighted_outputs = [out * weight for out, weight in zip(outputs, weights)]
            return weighted_outputs
        return output