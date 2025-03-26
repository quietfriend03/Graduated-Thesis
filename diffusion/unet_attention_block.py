import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention, CrossAttention

class UnetAttentionBlock():
    def __init__(self, n_heads: int, n_embedding: int, d_context: 768):
        super().__init__()
        channel = n_embedding * n_heads
        
        self.group_norm = nn.GroupNorm(32, channel, eps=1e-6)
        self.conv = nn.Conv2d(channel, d_context, kernel_size=1, padding=0)
        
        self.layer_norm_1 = nn.LayerNorm(channel)
        self.attention_1 = SelfAttention(n_heads, channel, in_proj_bias=False)
        self.layer_norm_2 = nn.LayerNorm(channel)
        self.attention_2 = CrossAttention(n_heads, channel, in_proj_bias=False)
        self.layer_norm_3 = nn.LayerNorm(channel)
        self.linear_geglu_1 = nn.Linear(channel, channel * 4 * 2)
        self.linear_geglu_2 = nn.Linear(4 * channel, channel)
        self.conv_output = nn.Conv2d(channel, channel, kernel_size=1, padding=0)
        
    def forward(self, x, context):
        residual_long = x
        
        x = self.group_norm(x)
        
        x = self.conv(x)
        
        batch, features, height, width = x.shape 
        
        # (Batch, Features, Height, Width) => (Batch, Features, Height * Width)
        x = x.view(batch, features, height * width)
        
        # (Batch, Features, Height * Width) => (Batch, Height * Width, Features)
        x = x.transpose(-1, -2)
        
        
        # Normalization + Self-Attention with skip connection
        residual_short = x
        
        x = self.layer_norm_1(x)
        self.attention_1(x)
        x += residual_short
        
        residual_short = x
        
        # Normalization + Cross-Attention with skip connection 
        x = self.layer_norm_2(x)
        
        # Cross Attention
        self.attention_2(x, context)
        
        x += residual_short
        
        residual_short = x
        
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        
        x = x * F.silu(gate)
        
        x = self.linear_geglu_2(x)
        
        x = x + residual_short
        
        # (Batch, Height * Width, Features) => (Batch, Features, Height * Width)
        x = x.transpose(-1, -2)
        
        x = x.view(batch, features, height, width)
        
        return self.conv_output(x) + residual_long
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        