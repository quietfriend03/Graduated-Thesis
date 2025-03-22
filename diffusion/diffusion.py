import torch
import torch.nn as nn
import torch.nn.functional as F
from time_embedding import TimeEmbedding
from unet_layer import UnetOutputLayer
from unet import UNET

class Diffusion(nn.Module):
    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.output = UnetOutputLayer(320, 4)
        
    def forward(self, latent: torch.Tensor, text: torch.Tensor, time: torch.Tensor):
        # LATENT EMBEDDING (4, Height/8, Width/8) 
        # TIME EMBEDDING (1, 360)
        
        # (1, 360) => (1, 1440)
        time = self.time_embedding(time)
        
        # (4, Height/8, Width/8) => (360, Height/8, Width/8)
        output = self.unet(latent, text, time)
        
        output = self.output(output)
        
        return output