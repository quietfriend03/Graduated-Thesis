import torch
import torch.nn as nn
import torch.nn.functional as F
import attention as SelfAttention
from module.attention_block import AttentionBlock
from module.residual_block import ResidualBlock


class Decoder(nn.Sequential):
    def __init__(self):
        super.__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
        
            ResidualBlock(512, 512),    
            AttentionBlock(512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            
            ResidualBlock(512, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            
            ResidualBlock(256, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, 4, Height, Width)
        x /= 0.18125
        
        for module in self:
            x = module(x)
            
        # (Batch, 3, Height, Width)
        return x
    