import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPLayer(nn.Module):
    def __init__(self, n_heads: int, n_embedding: int):
        super().__init__()
        self.norm_layer1 = nn.LayerNorm(n_embedding)
        self.attention = SelfAttention(n_heads, n_embedding)
        self.norm_layer2 = nn.LayerNorm(n_embedding)
        self.linear1 = nn.Linear(n_embedding, 4 * n_embedding)
        self.linear2 = nn.Linear(4 * n_embedding, n_embedding)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SELF-ATTENTION LAYER
        residual = x
        
        x = self.norm_layer1(x)
        x = self.attention(x, causal_mask=False)
        x += residual
        
        # FEED-FORWARD LAYER
        residual = x
        
        x = self.norm_layer2(x)
        x = F.gelu(self.linear1(x))
        x = self.linear2(x)
        x += residual
        
        return x
