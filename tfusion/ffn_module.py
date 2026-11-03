from mmengine.model import BaseModule
from mmengine.model.weight_init import xavier_init, constant_init
import torch, torch.nn as nn
import numpy as np
from .safe_ops import safe_sigmoid, safe_inverse_sigmoid, get_rotation_matrix


class FFN(BaseModule):
    def __init__(
        self,
        embed_dims):
        super(FFN, self).__init__()
        self.layers = nn.Sequential(nn.Linear(embed_dims, embed_dims * 4),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Linear(embed_dims * 4, embed_dims),
                                    nn.Dropout(0.1))
        self.layer_norm = nn.LayerNorm(embed_dims)

    def forward(self, instance_feature):
        identity = instance_feature
        out = self.layers(instance_feature)
        res = self.layer_norm(out + identity) # residual connection
        
        return res
