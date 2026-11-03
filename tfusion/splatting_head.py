import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmengine.runner.amp import autocast
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from .safe_ops import get_rotation_matrix, softmax_inverse_with_grad
import local_aggregate_t_prob_gs
import local_aggregate_t_prob_gs_occ3d
import local_aggregate_t_prob_gs_semanticKITTI
import local_aggregate_t_prob_sq
import local_aggregate_t_prob_sq_occ3d
import local_aggregate_t_prob_sq_semanticKITTI
import local_aggregate_t_prob_sq_iw
import local_aggregate_t_prob_sq_iw_occ3d
import local_aggregate_t_prob_sq_iw_semanticKITTI
from mmdet.models.utils.misc import multi_apply


@MODELS.register_module()
class SplattingHead(BaseModule):
    def __init__(self,
                scale_multiplier,
                H, 
                W, 
                D,
                pc_min,
                grid_size,
                use_sq=False,
                use_sq_iw=False,
                use_occ3d=False,
                use_semanticKITTI=False):
        super().__init__()
        self.use_sq = use_sq
        self.use_sq_iw = use_sq_iw
        self.use_occ3d = use_occ3d
        self.use_semanticKITTI = use_semanticKITTI
        if self.use_occ3d:
            if self.use_sq:
                self.aggregator = local_aggregate_t_prob_sq_occ3d.LocalAggregator(scale_multiplier=scale_multiplier,
                                                                                H=H, 
                                                                                W=W, 
                                                                                D=D,
                                                                                pc_min=pc_min,
                                                                                grid_size=grid_size)
            elif self.use_sq_iw:
                self.aggregator = local_aggregate_t_prob_sq_iw_occ3d.LocalAggregator(scale_multiplier=scale_multiplier,
                                                                                    H=H, 
                                                                                    W=W, 
                                                                                    D=D,
                                                                                    pc_min=pc_min,
                                                                                    grid_size=grid_size)
            else:
                self.aggregator = local_aggregate_t_prob_gs_occ3d.LocalAggregator(scale_multiplier=scale_multiplier,
                                                                                H=H, 
                                                                                W=W, 
                                                                                D=D,
                                                                                pc_min=pc_min,
                                                                                grid_size=grid_size)
        elif self.use_semanticKITTI:
            if self.use_sq:
                self.aggregator = local_aggregate_t_prob_sq_semanticKITTI.LocalAggregator(scale_multiplier=scale_multiplier,
                                                                                        H=H, 
                                                                                        W=W, 
                                                                                        D=D,
                                                                                        pc_min=pc_min,
                                                                                        grid_size=grid_size)
            elif self.use_sq_iw:
                self.aggregator = local_aggregate_t_prob_sq_iw_semanticKITTI.LocalAggregator(scale_multiplier=scale_multiplier,
                                                                                            H=H, 
                                                                                            W=W, 
                                                                                            D=D,
                                                                                            pc_min=pc_min,
                                                                                            grid_size=grid_size)
            else:
                self.aggregator = local_aggregate_t_prob_gs_semanticKITTI.LocalAggregator(scale_multiplier=scale_multiplier,
                                                                                        H=H, 
                                                                                        W=W, 
                                                                                        D=D,
                                                                                        pc_min=pc_min,
                                                                                        grid_size=grid_size)
        else:
            if self.use_sq:
                self.aggregator = local_aggregate_t_prob_sq.LocalAggregator(scale_multiplier=scale_multiplier,
                                                                        H=H, 
                                                                        W=W, 
                                                                        D=D,
                                                                        pc_min=pc_min,
                                                                        grid_size=grid_size)
            elif self.use_sq_iw:
                self.aggregator = local_aggregate_t_prob_sq_iw.LocalAggregator(scale_multiplier=scale_multiplier,
                                                                            H=H, 
                                                                            W=W, 
                                                                            D=D,
                                                                            pc_min=pc_min,
                                                                            grid_size=grid_size)
            else:
                self.aggregator = local_aggregate_t_prob_gs.LocalAggregator(scale_multiplier=scale_multiplier,
                                                                    H=H, 
                                                                    W=W, 
                                                                    D=D,
                                                                    pc_min=pc_min,
                                                                    grid_size=grid_size)
        
    def prepare_gaussian_args(self, scale, quaternion):
        bs, num_g = scale.shape[:2]
        S = torch.zeros(bs, num_g, 3, 3, dtype=scale.dtype, device=scale.device)
        S[..., 0, 0] = scale[..., 0]
        S[..., 1, 1] = scale[..., 1]
        S[..., 2, 2] = scale[..., 2]
        R = get_rotation_matrix(quaternion) # g, 3, 3
        M = torch.matmul(S, R)
        Cov = torch.matmul(M.transpose(-1, -2), M)
        CovInv = Cov.cpu().inverse().cuda() # g, 3, 3
        
        return CovInv
    
    def splatting_t_gs(self, sample_xyz, mean, opacity, semantic, scale, CovIns):        
        output = self.aggregator(sample_xyz.clone().float(),
                                 mean,
                                 opacity,
                                 semantic,
                                 scale,
                                 CovIns)   
        sem = output[0] * output[1].unsqueeze(-1)
        geo = 1 - output[1].unsqueeze(-1) # empty probability
        res = torch.cat([geo, sem], dim=-1)
        
        return res[None]
    
    def splatting_t_sq(self, sample_xyz, mean, opacity, u, v, semantic, scale, rot3D):        
        output = self.aggregator(sample_xyz.clone().float(),
                                 mean,
                                 opacity,
                                 u,
                                 v,
                                 semantic,
                                 scale,
                                 rot3D)   
        sem = output[0] * output[1].unsqueeze(-1)
        geo = 1 - output[1].unsqueeze(-1) # empty probability
        res = torch.cat([geo, sem], dim=-1)
        
        return res[None]
    
    def splatting_t_sq_iw(self, sample_xyz, mean, opacity, u, v, semantic, deform_weight, scale, rot3D):        
        output = self.aggregator(sample_xyz.clone().float(),
                                 mean,
                                 opacity,
                                 u,
                                 v,
                                 semantic,
                                 deform_weight,
                                 scale,
                                 rot3D)   
        sem = output[0] * output[1].unsqueeze(-1)
        geo = 1 - output[1].unsqueeze(-1) # empty probability
        res = torch.cat([geo, sem], dim=-1)
        
        return res[None]
    
    def forward(self,
                sample_loc,
                t_gs):
        t_means = t_gs.means
        t_scales = t_gs.scales
        t_quaternions = t_gs.rotations
        t_opas = t_gs.opacities
        t_u = t_gs.u
        t_v = t_gs.v
        t_dw = t_gs.deform_weights
        t_sem = t_gs.semantics
        if self.use_sq:
            t_rots = get_rotation_matrix(t_quaternions)
            out = multi_apply(self.splatting_t_sq, 
                            sample_loc[0], 
                            t_means, 
                            t_opas,
                            t_u,
                            t_v, 
                            t_sem,
                            t_scales, 
                            t_rots)[0]
        elif self.use_sq_iw:
            t_rots = get_rotation_matrix(t_quaternions)
            out = multi_apply(self.splatting_t_sq_iw, 
                            sample_loc[0], 
                            t_means, 
                            t_opas,
                            t_u,
                            t_v, 
                            t_sem,
                            t_dw, 
                            t_scales, 
                            t_rots)[0]
        else:
            CovInvs = self.prepare_gaussian_args(t_scales, t_quaternions)
            out = multi_apply(self.splatting_t_gs, 
                            sample_loc[0], 
                            t_means, 
                            t_opas, 
                            t_sem, 
                            t_scales, 
                            CovInvs)[0]
        out_res = torch.stack(out)
        
        return out_res