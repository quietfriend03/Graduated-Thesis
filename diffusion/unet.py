import torch
import torch.nn as nn
import torch.nn.functional as F
from unet_attention_block import UnetAttentionBlock
from unet_residual_block import UnetResidualBlock
class SwitchSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, text: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UnetResidualBlock):
                x = layer(x, time)
            elif isinstance(layer, UnetAttentionBlock):
                x = layer(x, text)
            else:
                x = layer(x)
        return x
    
class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # (Features, Height, Width) => (Features, Height * 2, Width * 2)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

class UNET():
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential([
            # (4, Height, Width) => (320, Height/8, Width/8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(UnetResidualBlock(320, 320), UnetAttentionBlock(8, 40)),
            SwitchSequential(UnetResidualBlock(320, 320), UnetAttentionBlock(8, 40)),
            
            # (320, Height/16, Width/16) => (640, Height/16, Width/16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UnetResidualBlock(320, 640), UnetAttentionBlock(8, 80)),
            SwitchSequential(UnetResidualBlock(640, 640), UnetAttentionBlock(8, 80)),
            
            # (640, Height/32, Width/32) => (1280, Height/32, Width/32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UnetResidualBlock(640, 1280), UnetAttentionBlock(8, 160)),
            SwitchSequential(UnetResidualBlock(1280, 1280), UnetAttentionBlock(8, 160)),
            
            # (1280, Height/64, Width/64) => (1280, Height/64, Width/64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UnetResidualBlock(1280, 1280)),
            SwitchSequential(UnetResidualBlock(1280, 1280)),
        ])
        
        self.bottleneck = nn.Sequential([
            UnetResidualBlock(1280, 1280),
            UnetAttentionBlock(8, 160),
            UnetAttentionBlock(1280, 1280),
        ])
        
        self.decoder = nn.Sequential([
            SwitchSequential(UnetResidualBlock(2560, 1280)),
            SwitchSequential(UnetResidualBlock(2560, 1280)),
            
            SwitchSequential(UnetResidualBlock(2560, 1280), Upsample(1280)),            
            SwitchSequential(UnetResidualBlock(2560, 1280), UnetAttentionBlock(8, 160)),
            SwitchSequential(UnetResidualBlock(2560, 1280), UnetAttentionBlock(8, 160)),
            
            SwitchSequential(UnetResidualBlock(1920, 1280), UnetAttentionBlock(8, 160), Upsample(1280)),
            SwitchSequential(UnetResidualBlock(1920, 640), UnetAttentionBlock(8, 80)), 
            SwitchSequential(UnetResidualBlock(1280, 640), UnetAttentionBlock(8, 80)), 
            
            SwitchSequential(UnetResidualBlock(960, 640), UnetAttentionBlock(8, 80), Upsample(640)),
            SwitchSequential(UnetResidualBlock(960, 320), UnetAttentionBlock(8, 40)),
            SwitchSequential(UnetResidualBlock(640, 320), UnetAttentionBlock(8, 40)),
            SwitchSequential(UnetResidualBlock(640, 320), UnetAttentionBlock(8, 40)),
        ])