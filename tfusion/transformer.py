import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from .safe_ops import safe_sigmoid, safe_inverse_sigmoid
from .self_atten import SelfAtten
from .cross_attention_3d import DeformableFeatureAggregation3D
from .ffn_module import FFN
from .anchor_endocer import AnchorEncoder
from .refine_module import RefineModule


@MODELS.register_module()
class Transformer(BaseModule):
    def __init__(self,
                 num_layers,
                 num_head,
                 num_levels,
                 num_cams,
                 embed_dims,
                 dbound,
                 grid_size_pol,
                 grid_size_occ,
                 pc_range,
                 pol_range,
                 kernel_size,
                 scale_range,
                 num_anchor,
                 splatting_head,
                 mul_sup=False,
                 use_sq=False,
                 use_sq_iw=False,
                 use_occ3d=False,
                 use_semanticKITTI=False):
        super().__init__()
        self.mul_sup = mul_sup
        self.use_sq = use_sq
        self.use_sq_iw = use_sq_iw
        self.use_occ3d = use_occ3d
        self.use_semanticKITTI = use_semanticKITTI
        self.num_layers = num_layers
        self.num_head = num_head
        self.num_levels = num_levels
        self.num_cams = num_cams
        self.embed_dims = embed_dims
        self.grid_size_pol = torch.tensor(grid_size_pol, dtype=torch.float32)
        self.grid_size_occ = torch.tensor(grid_size_occ)
        self.min_bound = torch.tensor(pc_range[:3], dtype=torch.float32)
        self.min_bound_pol = torch.tensor(pol_range[:3])
        self.max_bound = torch.tensor(pc_range[3:], dtype=torch.float32)
        self.max_bound_pol = torch.tensor(pol_range[3:])
        self.scale_range = torch.tensor(scale_range, dtype=torch.float32)
        self.intervals = (self.max_bound - self.min_bound) / self.grid_size_occ
        self.intervals_pol = (self.max_bound_pol - self.min_bound_pol) / self.grid_size_pol
        self.splatting_head = MODELS.build(splatting_head)
        self.anchor_encoder = AnchorEncoder(embed_dims=embed_dims,
                                            use_sq=self.use_sq,
                                            use_sq_iw=self.use_sq_iw,
                                            use_occ3d=self.use_occ3d,
                                            use_semanticKITTI=self.use_semanticKITTI)
        self.dbound = dbound
        self.depth_channels = int((self.dbound[1] - self.dbound[0]) // self.dbound[2])
        self.depth_aware_feat = nn.Sequential(nn.Linear(self.depth_channels, self.depth_channels//2),
                                              nn.GELU(),
                                              nn.Linear(self.depth_channels//2, embed_dims//num_head))
        self.self_attens = nn.ModuleList()
        self.cross_attens = nn.ModuleList() 
        self.FFNs_1 = nn.ModuleList()
        self.FFNs_2 = nn.ModuleList()
        self.refines = nn.ModuleList()
        for i in range(num_layers):
            self_atten = SelfAtten(embed_channels=embed_dims,
                                    polar_bound=pol_range,
                                    cart_bound=pc_range,
                                    grid_size=grid_size_pol,
                                    kernel_size=kernel_size)
            
            cross_atten = DeformableFeatureAggregation3D(embed_dims=embed_dims,
                                                         num_head=self.num_head,    # number of head
                                                         num_levels=self.num_levels,    # 3
                                                         num_cams=self.num_cams,
                                                         dbound=dbound,
                                                         pol_range=pol_range,
                                                         grid_size_pol=grid_size_pol,
                                                         use_occ3d=self.use_occ3d,
                                                         use_semanticKITTI=self.use_semanticKITTI)
            
            ffn_1 = FFN(embed_dims=embed_dims)
            ffn_2 = FFN(embed_dims=embed_dims)
            refine = RefineModule(embed_dims=embed_dims,
                                  pc_range=pc_range,
                                  scale_range=scale_range,
                                  unit_xyz=[2.0, 2.0, 0.5], # 4/10,4/10,1
                                  uv_range=[0.1, 2.0],
                                  use_sq=self.use_sq,
                                  use_sq_iw=self.use_sq_iw,
                                  use_occ3d=self.use_occ3d,
                                  use_semanticKITTI=self.use_semanticKITTI) 
            
            
            self.self_attens.append(self_atten)
            self.cross_attens.append(cross_atten)        
            self.FFNs_1.append(ffn_1)
            self.FFNs_2.append(ffn_2)
            self.refines.append(refine)  
            
        scale = torch.ones(num_anchor, 3, dtype=torch.float) * 0.5
        scale = safe_inverse_sigmoid(scale)

        rots = torch.zeros(num_anchor, 4, dtype=torch.float)
        rots[:, 0] = 1
        opacity = safe_inverse_sigmoid(0.5 * torch.ones((num_anchor, 1), dtype=torch.float))
        if self.use_occ3d:
            semantic = torch.randn(num_anchor, 17, dtype=torch.float)
        elif self.use_semanticKITTI:
            semantic = torch.randn(num_anchor, 19, dtype=torch.float)
        else:
            semantic = torch.randn(num_anchor, 16, dtype=torch.float)
        if self.use_sq:
            u = safe_inverse_sigmoid(0.5 * torch.ones(num_anchor, 1, dtype=torch.float))
            v = safe_inverse_sigmoid(0.5 * torch.ones(num_anchor, 1, dtype=torch.float))
            anchor = torch.cat([scale, rots, opacity, u, v, semantic], dim=-1) # scale (3), rots (quaternion 4), opacity (1), U (1), V (1), deform_weights (24), semantic (16/17/19)
        elif self.use_sq_iw:
            u = safe_inverse_sigmoid(0.5 * torch.ones(num_anchor, 1, dtype=torch.float))
            v = safe_inverse_sigmoid(0.5 * torch.ones(num_anchor, 1, dtype=torch.float))
            deform_weights = safe_inverse_sigmoid(0.5 * torch.ones(num_anchor, 24, dtype=torch.float))
            anchor = torch.cat([scale, rots, opacity, u, v, deform_weights, semantic], dim=-1) # scale (3), rots (quaternion 4), opacity (1), U (1), V (1), deform_weights (24), semantic (16/17/19)
        else:
            anchor = torch.cat([scale, rots, opacity, semantic], dim=-1) # scale (3), rots (quaternion 4), opacity (1) semantic (16/17/19)

        self.num_anchor = num_anchor
        self.anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float32), requires_grad=False) # No gradient requirement
        self.instance_feature = nn.Parameter(torch.zeros([num_anchor, self.embed_dims]), requires_grad=False) # No gradient requirement
    
    
    def get_downsampled_gt_depth(self, gt_depths, downsample_factor):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // downsample_factor, downsample_factor, W // downsample_factor, downsample_factor, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, downsample_factor * downsample_factor)
        gt_depths_tmp = torch.where(gt_depths == 0.0,1e5 * torch.ones_like(gt_depths), gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // downsample_factor, W // downsample_factor)

        gt_depths = (gt_depths - (self.dbound[0] - self.dbound[2])) / self.dbound[2]
        gt_depths = torch.where((gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0), gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),num_classes=self.depth_channels + 1).view(-1, self.depth_channels + 1)[:, 1:]
        gt_depths = gt_depths.reshape(B, N, H // downsample_factor, W // downsample_factor, self.depth_channels)
        gt_depths = gt_depths.float()
        
        return gt_depths
    
    
    def get_spatial_shape_3D(self, spatial_shape, depth_dim):
        spatial_shape_depth = spatial_shape.new_ones(*spatial_shape.shape[:-1], 1) * depth_dim
        spatial_shape_3D = torch.cat([spatial_shape, spatial_shape_depth], dim=-1)
        return spatial_shape_3D.contiguous()
    
    
    def forward(self,
                vol_lidar,
                sample_loc,
                anchor_coor,
                cam_logits,
                vis_feats,
                depth_map,
                img_metas):
        bs = anchor_coor.shape[0]
        anchor = torch.cat([anchor_coor, torch.tile(self.anchor[None], (bs, 1, 1))], dim=-1) # logits cart_xyz, scale, rots, opacity, semantic
        instance_feature = torch.tile(self.instance_feature[None], (bs, 1, 1))
        anchor_embed = self.anchor_encoder(anchor)
        ###### Get Flattened visual features ########
        feat_flatten = []
        spatial_shapes = []
        for feat in vis_feats:
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3, 4).permute(0, 1, 3, 2).float()
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)
        feat_flatten = torch.cat(feat_flatten, dim=2)
        feat_flatten = feat_flatten.view(bs, self.num_cams, feat_flatten.shape[2], self.num_head, -1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=instance_feature.device)
        spatial_shapes_3D = self.get_spatial_shape_3D(spatial_shapes, self.depth_channels)
        spatial_shapes_3D = spatial_shapes_3D[None].repeat(bs, 1, 1)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        level_start_index = level_start_index[None].repeat(bs,1)
        ##############################################
        
        ###### Only generate lidar-side depth_dist once ###########
        depth_lvl0 = self.get_downsampled_gt_depth(depth_map, 8)
        depth_lvl1 = self.get_downsampled_gt_depth(depth_map, 16)
        depth_lvl2 = self.get_downsampled_gt_depth(depth_map, 32)
        lidar_depth_dist = torch.cat([depth_lvl0.flatten(2,3), 
                                      depth_lvl1.flatten(2,3), 
                                      depth_lvl2.flatten(2,3)], 
                                     dim=2)
        ########################################################
        
        ###### Only generate camera-side depth_dist once ###########
        cam_log = cam_logits[..., :-1].permute(0,1,4,2,3)
        B, num_cam, C, H, W = cam_log.shape
        cam_log = cam_log.reshape(B*num_cam,C,H,W)
        cam_depth = [cam_log]
        for i in range(spatial_shapes_3D.size(1)-1):
            cam_depth_dist = F.interpolate(cam_depth[-1], scale_factor=0.5, mode='bilinear', align_corners=False, antialias=True) # mode="nearest"
            cam_depth.append(cam_depth_dist)
        cam_depth = torch.cat([depth.flatten(2,3).permute(0,2,1) for depth in cam_depth], dim=1)
        cam_depth = cam_depth.reshape(B, num_cam, -1, C)
        depth_dist = safe_sigmoid(cam_depth)
        depth_dist = depth_dist + lidar_depth_dist
        depth_dist = torch.clamp(depth_dist, 0.0, 1.0)
        depth_dist_out = depth_dist[:, :, :, None, :].repeat(1,1,1,self.num_head,1) # repeat num of head times
        depth_feat = safe_inverse_sigmoid(depth_dist)
        depth_feat = self.depth_aware_feat(depth_feat)
        feat_flatten_depth_aware = feat_flatten + depth_feat[:, :, :, None, :].repeat(1,1,1,self.num_head,1)
        ############################################################
        t_primitives = []
        for cross_atten, FFN_1, self_atten, FFN_2, refine in zip(self.cross_attens, self.FFNs_1, self.self_attens, self.FFNs_2, self.refines):
            instance_feature = cross_atten(vol_lidar,
                                           anchor,
                                           instance_feature,
                                           anchor_embed,
                                           feat_flatten_depth_aware,
                                           spatial_shapes_3D,
                                           level_start_index,
                                           depth_dist_out,
                                           img_metas)
            instance_feature = FFN_1(instance_feature)
            instance_feature = self_atten(instance_feature, anchor_embed, anchor)
            instance_feature = FFN_2(instance_feature)
            anchor, t_prim = refine(instance_feature, anchor_embed, anchor) 
            anchor_embed = self.anchor_encoder(anchor)
            t_primitives.append(t_prim)
            
        if self.training and self.mul_sup:
            output = []
            for t_p in t_primitives:
                res = self.splatting_head(sample_loc, t_p)
                output.append(res)
            
            return output
        
        else: 
            res = self.splatting_head(sample_loc, t_primitives[-1])
            
            return res
            
            
            