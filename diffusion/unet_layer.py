import torch
import torch.nn as nn
import torch.nn.functional as F

class UnetOutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        # x: (Batch, 360, Height/8, Width/8)
        
        x = self.group_norm(x)
        
        x = F.silu(x)
                
        x = self.conv(x)
        
        # (Batch, 4, Height/8, Width/8) => (Batch, 4, Height/8, Width/8)
        return x