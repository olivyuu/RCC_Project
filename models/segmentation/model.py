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
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, 
                                      kernel_size=2, stride=2)
        self.reduce_skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.attention_gate = AttentionGate(F_g=out_channels, F_l=out_channels, F_int=out_channels//2)
        
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
            
        self.expand_concat = nn.Conv3d(out_channels * 2, concat_channels, kernel_size=1)
        self.conv_block = nn.Sequential(
            ConvDropoutNormNonlin(concat_channels, out_channels),
            ConvDropoutNormNonlin(out_channels, out_channels)
        )
        self.channel_attention = ChannelAttention(out_channels)
        self.spatial_attention = SpatialAttention(out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, skip.shape[2:])
        skip = self.reduce_skip(skip)
        skip = self.attention_gate(x, skip)
        x = torch.cat((skip, x), dim=1)
        x = self.expand_concat(x)
        x = self.conv_block(x)
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
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

class SegmentationModel(nn.Module):
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
        if hasattr(self, layer_name):
            target_layer = getattr(self, layer_name)
            target_layer.register_forward_hook(self._get_activation)
            target_layer.register_full_backward_hook(self._get_gradient)
            self.target_layer = target_layer

    def forward(self, x):
        # Get normalized progressive weights
        if self.training:
            weights = self.softmax(self.progressive_weights)
        
        # Encoder
        skip_connections = []
        for i, down_block in enumerate(self.down_blocks[:-1]):
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
                deep_out = self.deep_supervision[i](x)
                deep_outputs.append(deep_out)

        output = self.final_conv(x)

        if self.training:
            outputs = [output] + deep_outputs
            weighted_outputs = [out * weight for out, weight in zip(outputs, weights)]
            return weighted_outputs
        return output