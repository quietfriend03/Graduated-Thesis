import torch
import torch.nn as nn
import torch.nn.functional as F

class Diffusion(nn.Module):
    def __init__(self):
        self.time_embedding = TimeEmbedding(360)
        self.unet = UNET()
        self.output = UNET_OutputLayer(360, 4)
        
    def forward(self, latent: torch.Tensor, text: torch.Tensor, time: torch.Tensor):
        # LATENT EMBEDDING (4, Height/8, Width/8) 
        # TIME EMBEDDING (1, 360)
        
        # (1, 360) => (1, 1440)
        time = self.time_embedding(time)
        
        # (4, Height/8, Width/8) => (360, Height/8, Width/8)
        output = self.unet(latent, text, time)
        
        output = self.output(output)
        
        return output