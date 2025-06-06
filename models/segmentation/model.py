import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Double convolution block"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class SegmentationModel(nn.Module):
    """
    U-Net style segmentation model optimized for single-channel CT input.
    
    Phase 1 configuration:
    - Single channel input (CT only)
    - Single channel output (tumor probability)
    - No assumption about kidney masks
    """
    def __init__(self, 
                in_channels: int = 1,  # Default to single channel (CT)
                out_channels: int = 1,  # Single channel output (tumor)
                features: int = 32):    # Base feature channels
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        
        # Verify valid input configuration
        if in_channels != 1:
            print(f"Warning: Expected in_channels=1 for Phase 1 (CT only), got {in_channels}")
        
        # Encoder (downsampling path)
        self.enc1 = DoubleConv(in_channels, features)
        self.enc2 = DoubleConv(features, features * 2)
        self.enc3 = DoubleConv(features * 2, features * 4)
        self.enc4 = DoubleConv(features * 4, features * 8)
        
        # Bottleneck
        self.bottleneck = DoubleConv(features * 8, features * 16)
        
        # Decoder (upsampling path)
        self.up4 = DoubleConv(features * 16, features * 8)
        self.up3 = DoubleConv(features * 8, features * 4)
        self.up2 = DoubleConv(features * 4, features * 2)
        self.up1 = DoubleConv(features * 2, features)
        
        # Final 1x1 convolution to get output channels
        self.final_conv = nn.Conv3d(features, out_channels, kernel_size=1)
        
        # Max pooling for downsampling
        self.pool = nn.MaxPool3d(2)
        
        # Print model configuration
        self._print_config()
        
    def _print_config(self):
        """Print model configuration details"""
        print("\nSegmentation Model Configuration:")
        print(f"Input channels: {self.in_channels} (CT only)")
        print(f"Output channels: {self.out_channels} (tumor probability)")
        print(f"Base features: {self.features}")
        print(f"Encoder depths: {self.features} → {self.features*2} → {self.features*4} → {self.features*8}")
        print(f"Bottleneck features: {self.features*16}")
        
        # Calculate approximate number of parameters
        num_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {num_params:,}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net
        
        Args:
            x: Input tensor [B, 1, D, H, W] (single channel CT)
            
        Returns:
            Output tensor [B, 1, D, H, W] (tumor probability logits)
        """
        # Input validation
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected input with {self.in_channels} channels, got {x.shape[1]} channels.\n"
                f"Input shape: {tuple(x.shape)}"
            )
        
        # Save input spatial dimensions for size validation
        input_size = x.shape[2:]
        
        # Encoder path with skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder path with skip connections
        # Interpolate bottleneck up to e4 size
        d4 = self.up4(torch.cat([
            F.interpolate(b, size=e4.shape[2:], mode='trilinear', align_corners=False),
            e4
        ], dim=1))
        
        # Continue upsampling and concatenating
        d3 = self.up3(torch.cat([
            F.interpolate(d4, size=e3.shape[2:], mode='trilinear', align_corners=False),
            e3
        ], dim=1))
        
        d2 = self.up2(torch.cat([
            F.interpolate(d3, size=e2.shape[2:], mode='trilinear', align_corners=False),
            e2
        ], dim=1))
        
        d1 = self.up1(torch.cat([
            F.interpolate(d2, size=e1.shape[2:], mode='trilinear', align_corners=False),
            e1
        ], dim=1))
        
        # Final convolution
        out = self.final_conv(d1)
        
        # Verify output size matches input
        if out.shape[2:] != input_size:
            out = F.interpolate(
                out, 
                size=input_size,
                mode='trilinear',
                align_corners=False
            )
        
        return out