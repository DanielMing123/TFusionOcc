from mmengine.model import BaseModule
from mmengine.model.weight_init import xavier_init, constant_init
import torch, torch.nn as nn
import numpy as np
from .safe_ops import safe_sigmoid, safe_inverse_sigmoid, get_rotation_matrix, PrimitivePrediction


class RefineModule(BaseModule):
    def __init__(self,
                 embed_dims,
                 pc_range,
                 scale_range,
                 unit_xyz,
                 uv_range=[0.1, 2],
                 use_sq=False,
                 use_sq_iw=False,
                 use_occ3d=False,
                 use_semanticKITTI=False):
        super().__init__()
        self.use_sq = use_sq
        self.use_sq_iw = use_sq_iw
        self.use_occ3d = use_occ3d
        self.use_semanticKITTI = use_semanticKITTI
        self.embed_dims = embed_dims
        self.pc_range = torch.tensor(pc_range, dtype=torch.float)
        self.scale_range = torch.tensor(scale_range, dtype=torch.float)
        self.unit_xyz = torch.tensor(unit_xyz, dtype=torch.float)
        self.uv_range = torch.tensor(uv_range, dtype=torch.float)
        if self.use_occ3d:
            if self.use_sq:
                self.layers = nn.Sequential(nn.Linear(embed_dims, embed_dims),
                                            nn.ReLU(),
                                            nn.Linear(embed_dims, embed_dims),
                                            nn.ReLU(),
                                            nn.LayerNorm(embed_dims),
                                            nn.Linear(embed_dims, embed_dims),
                                            nn.ReLU(),
                                            nn.Linear(embed_dims, embed_dims),
                                            nn.ReLU(),
                                            nn.LayerNorm(embed_dims),
                                            nn.Linear(embed_dims, 30))
            elif self.use_sq_iw:
                self.layers = nn.Sequential(nn.Linear(embed_dims, embed_dims),
                                            nn.ReLU(),
                                            nn.Linear(embed_dims, embed_dims),
                                            nn.ReLU(),
                                            nn.LayerNorm(embed_dims),
                                            nn.Linear(embed_dims, embed_dims),
                                            nn.ReLU(),
                                            nn.Linear(embed_dims, embed_dims),
                                            nn.ReLU(),
                                            nn.LayerNorm(embed_dims),
                                            nn.Linear(embed_dims, 54))
            else:
                self.layers = nn.Sequential(nn.Linear(embed_dims, embed_dims),
                                            nn.ReLU(),
                                            nn.Linear(embed_dims, embed_dims),
                                            nn.ReLU(),
                                            nn.LayerNorm(embed_dims),
                                            nn.Linear(embed_dims, embed_dims),
                                            nn.ReLU(),
                                            nn.Linear(embed_dims, embed_dims),
                                            nn.ReLU(),
                                            nn.LayerNorm(embed_dims),
                                            nn.Linear(embed_dims, 28))
        elif self.use_semanticKITTI:
            if self.use_sq:
                self.layers = nn.Sequential(nn.Linear(embed_dims, embed_dims),
                                            nn.ReLU(),
                                            nn.Linear(embed_dims, embed_dims),
                                            nn.ReLU(),
                                            nn.LayerNorm(embed_dims),
                                            nn.Linear(embed_dims, embed_dims),
                                            nn.ReLU(),
                                            nn.Linear(embed_dims, embed_dims),
                                            nn.ReLU(),
                                            nn.LayerNorm(embed_dims),
                                            nn.Linear(embed_dims, 32))
            elif self.use_sq_iw:
                self.layers = nn.Sequential(nn.Linear(embed_dims, embed_dims),
                                            nn.ReLU(),
                                            nn.Linear(embed_dims, embed_dims),
                                            nn.ReLU(),
                                            nn.LayerNorm(embed_dims),
                                            nn.Linear(embed_dims, embed_dims),
                                            nn.ReLU(),
                                            nn.Linear(embed_dims, embed_dims),
                                            nn.ReLU(),
                                            nn.LayerNorm(embed_dims),
                                            nn.Linear(embed_dims, 56))
            else:
                self.layers = nn.Sequential(nn.Linear(embed_dims, embed_dims),
                                            nn.ReLU(),
                                            nn.Linear(embed_dims, embed_dims),
                                            nn.ReLU(),
                                            nn.LayerNorm(embed_dims),
                                            nn.Linear(embed_dims, embed_dims),
                                            nn.ReLU(),
                                            nn.Linear(embed_dims, embed_dims),
                                            nn.ReLU(),
                                            nn.LayerNorm(embed_dims),
                                            nn.Linear(embed_dims, 30))
        else:
            if self.use_sq:
                self.layers = nn.Sequential(nn.Linear(embed_dims, embed_dims),
                                            nn.ReLU(),
                                            nn.Linear(embed_dims, embed_dims),
                                            nn.ReLU(),
                                            nn.LayerNorm(embed_dims),
                                            nn.Linear(embed_dims, embed_dims),
                                            nn.ReLU(),
                                            nn.Linear(embed_dims, embed_dims),
                                            nn.ReLU(),
                                            nn.LayerNorm(embed_dims),
                                            nn.Linear(embed_dims, 29))
            elif self.use_sq_iw:
                self.layers = nn.Sequential(nn.Linear(embed_dims, embed_dims),
                                            nn.ReLU(),
                                            nn.Linear(embed_dims, embed_dims),
                                            nn.ReLU(),
                                            nn.LayerNorm(embed_dims),
                                            nn.Linear(embed_dims, embed_dims),
                                            nn.ReLU(),
                                            nn.Linear(embed_dims, embed_dims),
                                            nn.ReLU(),
                                            nn.LayerNorm(embed_dims),
                                            nn.Linear(embed_dims, 53))
            else:
                self.layers = nn.Sequential(nn.Linear(embed_dims, embed_dims),
                                            nn.ReLU(),
                                            nn.Linear(embed_dims, embed_dims),
                                            nn.ReLU(),
                                            nn.LayerNorm(embed_dims),
                                            nn.Linear(embed_dims, embed_dims),
                                            nn.ReLU(),
                                            nn.Linear(embed_dims, embed_dims),
                                            nn.ReLU(),
                                            nn.LayerNorm(embed_dims),
                                            nn.Linear(embed_dims, 27))
    
    def forward(self,
                instance_feature,
                anchor_embed,
                anchor): 
        
        device = instance_feature.device
        
        output = self.layers(instance_feature + anchor_embed) 

        #### for xyz
        delta_xyz = (2 * safe_sigmoid(output[..., :3]) - 1.) * self.unit_xyz[None, None].to(device)
        original_xyz = safe_sigmoid(anchor[..., :3]) * (self.pc_range[3:].to(device) - self.pc_range[:3].to(device)) + self.pc_range[:3].to(device)
        anchor_xyz = original_xyz + delta_xyz
        anchor_xyz = safe_inverse_sigmoid((anchor_xyz - self.pc_range[:3].to(device)) / (self.pc_range[3:].to(device) - self.pc_range[:3].to(device)))

        #### for scale
        anchor_scale = output[..., 3:6]

        #### for rotation
        anchor_rotation = output[..., 6:10]
        anchor_rotation = torch.nn.functional.normalize(anchor_rotation, 2, -1)

        #### for opacity
        anchor_opa = output[..., 10:11]


        if self.use_sq:
            u = output[..., 11:12]
            v = output[..., 12:13]
            #### for semantic
            anchor_sem = output[..., 13:]
            output = torch.cat([anchor_xyz, anchor_scale, anchor_rotation, anchor_opa, u, v, anchor_sem], dim=-1)
        elif self.use_sq_iw:
            u = output[..., 11:12]
            v = output[..., 12:13]
            deform_weights = output[..., 13:37]
            #### for semantic
            anchor_sem = output[..., 37:]
            output = torch.cat([anchor_xyz, anchor_scale, anchor_rotation, anchor_opa, u, v, deform_weights, anchor_sem], dim=-1)
        else:
            #### for semantic
            anchor_sem = output[..., 11:]
            output = torch.cat([anchor_xyz, anchor_scale, anchor_rotation, anchor_opa, anchor_sem], dim=-1)
        
        xyz = safe_sigmoid(anchor_xyz) * (self.pc_range[3:].to(device) - self.pc_range[:3].to(device)) + self.pc_range[:3].to(device) 

        scale = safe_sigmoid(anchor_scale)
        scale = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * scale
        
        semantics = anchor_sem.softmax(dim=-1)
        
        if self.use_sq:
            u_epsilon2 = self.uv_range[0] + (self.uv_range[1] - self.uv_range[0]) * safe_sigmoid(u)
            v_epsilon1 = self.uv_range[0] + (self.uv_range[1] - self.uv_range[0]) * safe_sigmoid(v)
            primitive = PrimitivePrediction(
                means=xyz,
                scales=scale,
                rotations=anchor_rotation,
                opacities=safe_sigmoid(anchor_opa),
                u=u_epsilon2,
                v=v_epsilon1,
                semantics=semantics)
        elif self.use_sq_iw:
            u_epsilon2 = self.uv_range[0] + (self.uv_range[1] - self.uv_range[0]) * safe_sigmoid(u)
            v_epsilon1 = self.uv_range[0] + (self.uv_range[1] - self.uv_range[0]) * safe_sigmoid(v)
            dw = torch.tanh(deform_weights) # self.s[None, None] * 
            primitive = PrimitivePrediction(
                means=xyz,
                scales=scale,
                rotations=anchor_rotation,
                opacities=safe_sigmoid(anchor_opa),
                u=u_epsilon2,
                v=v_epsilon1,
                deform_weights=dw,
                semantics=semantics)
        else:
            primitive = PrimitivePrediction(
                means=xyz,
                scales=scale,
                rotations=anchor_rotation,
                opacities=safe_sigmoid(anchor_opa),
                semantics=semantics)
            
        return output, primitive 