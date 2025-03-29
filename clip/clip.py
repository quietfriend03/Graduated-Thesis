import torch
from torch import nn
from torch.nn import functional as F
from clip.clip_embedding import CLIPTextEmbedding
from clip.clip_layer import CLIPLayer

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding  = CLIPTextEmbedding(50000, 768, 77)
        self.layer = nn.ModuleList([
            CLIPLayer(12, 768) for _ in range(12)
        ])
        self.norm_layer = nn.LayerNorm(768)
        
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        state = self.embedding(tokens)
        for layer in self.layer:
            state = layer(state)
        output = self.norm_layer(state)
        
        return output
        