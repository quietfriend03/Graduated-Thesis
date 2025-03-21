import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPTextEmbedding(nn.Module):
    def __init__(self, vocab_size: int, n_embedding: int, n_tokens: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embedding)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embedding))
        
    def forward(self, token):
        x = self.token_embedding(token)
        x += self.position_embedding
        return x