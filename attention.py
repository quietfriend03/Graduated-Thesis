import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embedding: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        
        self.in_proj = nn.Linear(d_embedding, 3 * d_embedding, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embedding, d_embedding, bias=out_proj_bias)
        
        self.n_heads = n_heads
        self.d_heads = d_embedding // n_heads
        
    def forward(self, x: torch.Tensor, causal_mask=False) -> torch.Tensor:
        # x: (Batch, Sequence Length, Dimension)
        input_shape = x.shape
        batch_size, sequence_length, dimension = input_shape
        
        intermim_shape = (batch_size, sequence_length, self.n_heads, self.d_heads)
        
        # (Batch, Sequence Length, Dimension) => (Batch, Sequence Length, 3 * Dimension) => 3 tensor shape (Batch, Sequence Length, Dimension)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        # (Batch, Sequence Length, Dimension) => (Batch, Sequence Length, Head(n_heads), Dimesion/Head(d_heads)) => (Batch, Head(n_heads), Sequence Length, Dimesion/Head(d_heads))
        q = q.view(intermim_shape).transpose(1, 2)
        k = k.view(intermim_shape).transpose(1, 2)
        v = v.view(intermim_shape).transpose(1, 2)
        
        weights = q @ k.transpose(-1, -2)
        if causal_mask:
            mask = torch.ones_like(weights, dtype=torch.bool).triu(1) 
            weights.masked_fill_(mask, -torch.inf)
            
        weights /= math.sqrt(self.d_heads)
        weights = F.softmax(weights, dim=-1)
        
        output = weights @ v
        
        output = output.transpose(1, 2).reshape(batch_size, sequence_length, dimension)
        
        output = self.out_proj(output)
        
        return output
        