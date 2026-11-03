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



class AnchorEncoder(BaseModule):
    def __init__(
        self, 
        embed_dims,
        use_sq=False,
        use_sq_iw=False,
        use_occ3d=False,
        use_semanticKITTI=False,
    ):
        super().__init__()
        self.use_sq = use_sq
        self.use_sq_iw = use_sq_iw
        self.use_occ3d = use_occ3d
        self.use_semanticKITTI = use_semanticKITTI
        self.xyz_fc = nn.Sequential(nn.Linear(3, embed_dims),
                                    nn.ReLU(),
                                    nn.LayerNorm(embed_dims),
                                    nn.Linear(embed_dims, embed_dims),
                                    nn.ReLU(),
                                    nn.LayerNorm(embed_dims))
        self.scale_fc = nn.Sequential(nn.Linear(3, embed_dims),
                                    nn.ReLU(),
                                    nn.LayerNorm(embed_dims),
                                    nn.Linear(embed_dims, embed_dims),
                                    nn.ReLU(),
                                    nn.LayerNorm(embed_dims))
        self.rot_fc = nn.Sequential(nn.Linear(4, embed_dims),
                                    nn.ReLU(),
                                    nn.LayerNorm(embed_dims),
                                    nn.Linear(embed_dims, embed_dims),
                                    nn.ReLU(),
                                    nn.LayerNorm(embed_dims))
        self.opa_fc = nn.Sequential(nn.Linear(1, embed_dims),
                                    nn.ReLU(),
                                    nn.LayerNorm(embed_dims),
                                    nn.Linear(embed_dims, embed_dims),
                                    nn.ReLU(),
                                    nn.LayerNorm(embed_dims))
        if self.use_sq:
            self.u_fc = nn.Sequential(nn.Linear(1, embed_dims),
                                      nn.ReLU(),
                                      nn.LayerNorm(embed_dims),
                                      nn.Linear(embed_dims, embed_dims),
                                      nn.ReLU(),
                                      nn.LayerNorm(embed_dims))
            self.v_fc = nn.Sequential(nn.Linear(1, embed_dims),
                                      nn.ReLU(),
                                      nn.LayerNorm(embed_dims),
                                      nn.Linear(embed_dims, embed_dims),
                                      nn.ReLU(),
                                      nn.LayerNorm(embed_dims))
        elif self.use_sq_iw:
            self.u_fc = nn.Sequential(nn.Linear(1, embed_dims),
                                      nn.ReLU(),
                                      nn.LayerNorm(embed_dims),
                                      nn.Linear(embed_dims, embed_dims),
                                      nn.ReLU(),
                                      nn.LayerNorm(embed_dims))
            self.v_fc = nn.Sequential(nn.Linear(1, embed_dims),
                                      nn.ReLU(),
                                      nn.LayerNorm(embed_dims),
                                      nn.Linear(embed_dims, embed_dims),
                                      nn.ReLU(),
                                      nn.LayerNorm(embed_dims))
            self.dw_fc = nn.Sequential(nn.Linear(24, embed_dims),
                                       nn.ReLU(),
                                       nn.LayerNorm(embed_dims),
                                       nn.Linear(embed_dims, embed_dims),
                                       nn.ReLU(),
                                       nn.LayerNorm(embed_dims))
        if self.use_occ3d:
            self.sem_fc = nn.Sequential(nn.Linear(17, embed_dims),
                                        nn.ReLU(),
                                        nn.LayerNorm(embed_dims),
                                        nn.Linear(embed_dims, embed_dims),
                                        nn.ReLU(),
                                        nn.LayerNorm(embed_dims))
        elif self.use_semanticKITTI:
            self.sem_fc = nn.Sequential(nn.Linear(19, embed_dims),
                                        nn.ReLU(),
                                        nn.LayerNorm(embed_dims),
                                        nn.Linear(embed_dims, embed_dims),
                                        nn.ReLU(),
                                        nn.LayerNorm(embed_dims))
        else:
            self.sem_fc = nn.Sequential(nn.Linear(16, embed_dims),
                                        nn.ReLU(),
                                        nn.LayerNorm(embed_dims),
                                        nn.Linear(embed_dims, embed_dims),
                                        nn.ReLU(),
                                        nn.LayerNorm(embed_dims))
            
        self.output_fc = nn.Sequential(nn.Linear(embed_dims, embed_dims),
                                        nn.ReLU(),
                                        nn.LayerNorm(embed_dims),
                                        nn.Linear(embed_dims, embed_dims),
                                        nn.ReLU(),
                                        nn.LayerNorm(embed_dims))
    
    def forward(self, anchor):
        xyz_feat = self.xyz_fc(anchor[..., :3])
        scale_feat = self.scale_fc(anchor[..., 3:6])
        rot_feat = self.rot_fc(anchor[..., 6:10])
        opacity_feat = self.opa_fc(anchor[..., 10:11])
        if self.use_sq:
            u_feat = self.u_fc(anchor[..., 11:12])
            v_feat = self.v_fc(anchor[..., 12:13])
            semantic_feat = self.sem_fc(anchor[..., 13:])
            output = xyz_feat + scale_feat + rot_feat + opacity_feat + u_feat + v_feat + semantic_feat
        elif self.use_sq_iw:
            u_feat = self.u_fc(anchor[..., 11:12])
            v_feat = self.v_fc(anchor[..., 12:13])
            dw_feat = self.dw_fc(anchor[..., 13:37])
            semantic_feat = self.sem_fc(anchor[..., 37:])
            output = xyz_feat + scale_feat + rot_feat + opacity_feat + u_feat + v_feat + dw_feat + semantic_feat
        else:
            semantic_feat = self.sem_fc(anchor[..., 11:])
            output = xyz_feat + scale_feat + rot_feat + opacity_feat + semantic_feat
        
        output = self.output_fc(output)
        
        return output