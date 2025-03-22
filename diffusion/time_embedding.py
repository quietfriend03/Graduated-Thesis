import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    def __init__(self, n_embedding: int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embedding, 4 * n_embedding)
        self.linear_2 = nn.Linear(4 * n_embedding, 4 * n_embedding)
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (1, 320)
        x = self.linear_1(x)
        
        x = F.silu(x)
        
        x = self.linear_2(x)
        
        # (1, 1280)
        return x