import torch
import torch.nn as nn
import torch.nn.functional as F

class UnetResidualBlock():
    def __init__(self, in_channels: int, out_channels: int, n_times: 1440):
        self.group_norm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_times, out_channels)
        
        self.group_norm_merge = nn.GroupNorm(32, out_channels)
        self.conv_merge = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            
    def forward(self, feature, time):
        residual = feature
        
        feature = self.group_norm_feature(feature)
        
        feature = F.silu(feature)
        
        feature = self.conv_feature(feature)
        
        time = F.silu(time)
        
        time = self.linear_time(time)
        
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        
        merged = self.group_norm_merge(merged)
        
        return merged + self.residual_layer(residual)