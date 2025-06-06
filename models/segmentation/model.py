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
        
        self.in_channels = int(in_channels)  # Ensure integer
        self.out_channels = int(out_channels)  # Ensure integer
        self.features = int(features)  # Ensure integer
        
        # Verify valid input configuration
        if self.in_channels != 1:
            print(f"Warning: Expected in_channels=1 for Phase 1 (CT only), got {self.in_channels}")
        
        # Encoder path
        self.enc1 = DoubleConv(self.in_channels, self.features)  # 32
        self.enc2 = DoubleConv(self.features, self.features * 2)  # 64
        self.enc3 = DoubleConv(self.features * 2, self.features * 4)  # 128
        self.enc4 = DoubleConv(self.features * 4, self.features * 8)  # 256
        
        # Bottleneck
        self.bottleneck = DoubleConv(self.features * 8, self.features * 16)  # 512
        
        # Decoder path with skip connections (reduce channels after concat)
        # After concat: 512 + 256 = 768 -> reduce to 256
        self.up4 = nn.Sequential(
            DoubleConv(self.features * 24, self.features * 8),  # 768 -> 256
        )
        
        # After concat: 256 + 128 = 384 -> reduce to 128
        self.up3 = nn.Sequential(
            DoubleConv(self.features * 12, self.features * 4),  # 384 -> 128
        )
        
        # After concat: 128 + 64 = 192 -> reduce to 64
        self.up2 = nn.Sequential(
            DoubleConv(self.features * 6, self.features * 2),  # 192 -> 64
        )
        
        # After concat: 64 + 32 = 96 -> reduce to 32
        self.up1 = nn.Sequential(
            DoubleConv(self.features * 3, self.features),  # 96 -> 32
        )
        
        # Final 1x1 convolution
        self.final_conv = nn.Conv3d(self.features, self.out_channels, kernel_size=1)
        
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
        e1 = self.enc1(x)  # 32
        e2 = self.enc2(self.pool(e1))  # 64
        e3 = self.enc3(self.pool(e2))  # 128
        e4 = self.enc4(self.pool(e3))  # 256
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))  # 512
        
        # Decoder path with skip connections
        # Upscale bottleneck and concat with e4: 512 + 256 = 768
        d4 = self.up4(torch.cat([
            F.interpolate(b, size=e4.shape[2:], mode='trilinear', align_corners=False),
            e4
        ], dim=1))  # 768 -> 256
        
        # Upscale d4 and concat with e3: 256 + 128 = 384
        d3 = self.up3(torch.cat([
            F.interpolate(d4, size=e3.shape[2:], mode='trilinear', align_corners=False),
            e3
        ], dim=1))  # 384 -> 128
        
        # Upscale d3 and concat with e2: 128 + 64 = 192
        d2 = self.up2(torch.cat([
            F.interpolate(d3, size=e2.shape[2:], mode='trilinear', align_corners=False),
            e2
        ], dim=1))  # 192 -> 64
        
        # Upscale d2 and concat with e1: 64 + 32 = 96
        d1 = self.up1(torch.cat([
            F.interpolate(d2, size=e1.shape[2:], mode='trilinear', align_corners=False),
            e1
        ], dim=1))  # 96 -> 32
        
        # Final convolution
        out = self.final_conv(d1)  # 32 -> 1
        
        # Verify output size matches input
        if out.shape[2:] != input_size:
            out = F.interpolate(
                out, 
                size=input_size,
                mode='trilinear',
                align_corners=False
            )
        
        return out