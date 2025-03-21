import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention

class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Features, Height, Width)
        residual = x
        
        x = self.group_norm(x)

        batch, features, height, width = x.shape
        
        # (Batch, Features, Height, Width) => (Batch, Features, Height * Width)
        x = x.view(batch, features, height * width)
        
        # (Batch, Features, Height * Width) => (Batch, Height * Width, Features)
        x = x.transpose(-1, -2)
        
        x = self.attention(x)
        
        # (Batch, Height * Width, Features) => (Batch, Features, Height * Width)
        x = x.transpose(-1, -2)
        
        # (Batch, Features, Height * Width) => (Batch, Features, Height, Width)
        x = x.view(batch, features, height, width)
        
        x += residual
        
        return x
    