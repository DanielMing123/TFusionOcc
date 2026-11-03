import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmengine.runner.amp import autocast
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
import spconv.pytorch as spconv
from spconv.pytorch import SparseConvTensor, SparseMaxPool3d
from mmengine.model.weight_init import xavier_init, constant_init
from .safe_ops import safe_sigmoid, safe_inverse_sigmoid


class SelfAtten(BaseModule):
    def __init__(
        self, 
        embed_channels,
        polar_bound,
        cart_bound,
        grid_size,
        kernel_size=5
    ):
        super().__init__()
        self.grid_size = torch.tensor(grid_size, dtype=torch.float32)
        self.polar_bound = torch.tensor(polar_bound)
        self.cart_bound = torch.tensor(cart_bound)
        self.intervals_pol = (self.polar_bound[3:] - self.polar_bound[:3]) / self.grid_size
        self.layer_norm = nn.LayerNorm(embed_channels)
        self.layer = spconv.SparseSequential(
            spconv.SubMConv3d(embed_channels, embed_channels, kernel_size, 1, (kernel_size - 1) // 2),
            nn.LayerNorm(embed_channels),
            nn.ReLU(True),
            spconv.SubMConv3d(embed_channels, embed_channels, kernel_size, 1, (kernel_size - 1) // 2),
            nn.LayerNorm(embed_channels),
            nn.ReLU(True),
            spconv.SubMConv3d(embed_channels, embed_channels, kernel_size, 1, (kernel_size - 1) // 2),
            nn.LayerNorm(embed_channels),
            nn.ReLU(True),
        )        
        
        self.output_proj = nn.Linear(embed_channels, embed_channels)
        
    
    def cart2polar(self, input_xyz):
        rho = torch.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
        phi = torch.arctan2(input_xyz[:, 1], input_xyz[:, 0])
        output_rdz =  torch.stack((rho, phi, input_xyz[:, 2]), axis=1)
        output_rdz = output_rdz.clamp(min=self.polar_bound[:3].to(output_rdz.device), max=self.polar_bound[3:].to(output_rdz.device))
        return output_rdz
        
    def forward(self, instance_feature, anchor_embed, anchor):
        # anchor: b, g, 11
        # instance_feature: b, g, c
        feature = instance_feature + anchor_embed
        identity = feature
        device = feature.device
        bs, g, _ = feature.shape

        # sparsify
        anchor_xyz = safe_sigmoid(anchor[...,:3])
        anchor_xyz = (anchor_xyz * (self.cart_bound[3:].to(device) - self.cart_bound[:3].to(device)) + self.cart_bound[:3].to(device)).flatten(0, 1)
        anchor_rdz = self.cart2polar(anchor_xyz)
        indices = (anchor_rdz - self.polar_bound[:3].to(device)) / self.intervals_pol.to(device)
        indices = indices.to(torch.int32)
        batched_indices = torch.cat([torch.arange(bs, device=indices.device, dtype=torch.int32).reshape(bs, 1, 1).expand(-1, g, -1).flatten(0, 1), indices], dim=-1)

        input = spconv.SparseConvTensor(
            feature.flatten(0, 1), # bg, c
            indices=batched_indices, # bg, 4
            spatial_shape=self.grid_size.to(torch.int32),
            batch_size=bs)

        output = self.layer(input)
        output = output.features.unflatten(0, (bs, g))
        res = self.output_proj(output)
        res_residual = res + identity
        out = self.layer_norm(res_residual)
        return out