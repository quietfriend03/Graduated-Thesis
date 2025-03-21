import torch
import torch.nn as nn
import torch.nn.functional as F
from module.attention_block import AttentionBlock
from module.residual_block import ResidualBlock

class Encoder(nn.Sequential):
    def __init__(self):
        super.__init__(
            #(Channel, Height, Width) => (128, Height, Width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            
            # (128, Height, Width) => (256, Height/2, Width/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            
            # (256, Height/2, Width/2) => (256, Height/4, Width/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),
            
            # (512, Height/4, Width/4) => (512, Height/8, Width/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            
            AttentionBlock(512),
            ResidualBlock(512, 512),
            
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            
            # (512, Height/4, Width/4) => (8, Height/8, Width/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            
            # (512, Height/4, Width/4) => (8, Height/8, Width/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )
    
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Channel, Height, Width)
        # noise: (Batch, Channel, Height/8, Width/8)
       
        for module in self:
            if getattr(module, "stride", None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
        #(8, Height/8, Width/8) => 2 tensor shape (4, Height/8, Width/8) and (4, Height/8, Width/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        
        #(4, Height/8, Width/8)
        log_variance = torch.clamp(log_variance, -20, 20)
        
        variance = torch.exp(log_variance)
        
        stdev = torch.sqrt(variance)
        
        x = mean + stdev * noise
        
        x *= 0.18125
        
        return x
        