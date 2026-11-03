from mmengine.model import BaseModule
from mmengine.model.weight_init import xavier_init, constant_init
import torch, torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .safe_ops import safe_sigmoid, safe_inverse_sigmoid, get_rotation_matrix
from .multi_scale_3ddeformable_attn_function import MultiScale3DDeformableAttnFunction_fp32
from mmdet.models.utils.misc import multi_apply


class SamplePointsGenerator(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        num_learnable_pts=0,
        learnable_fixed_scale=1,
        fix_scale=None,
        pc_range=None,
        scale_range=None
    ):
        super(SamplePointsGenerator, self).__init__()
        self.embed_dims = embed_dims
        self.num_learnable_pts = num_learnable_pts
        self.learnable_fixed_scale = learnable_fixed_scale
        if fix_scale is None:
            fix_scale = ((0.0, 0.0, 0.0),)
        self.fix_scale = np.array(fix_scale)
        self.num_pts = len(self.fix_scale) + num_learnable_pts
        if num_learnable_pts > 0:
            self.learnable_fc = nn.Linear(self.embed_dims, num_learnable_pts * 3)

        self.pc_range = pc_range
        self.scale_range = scale_range

    def init_weight(self):
        if self.num_learnable_pts > 0:
            xavier_init(self.learnable_fc, distribution="uniform", bias=0.0)

    def forward(self,
                anchor,
                instance_feature=None):
        
        bs, num_anchor = anchor.shape[:2]
        fix_scale = anchor.new_tensor(self.fix_scale)
        scale = fix_scale[None, None].tile([bs, num_anchor, 1, 1])
        if self.num_learnable_pts > 0 and instance_feature is not None:
            learnable_scale = (safe_sigmoid(self.learnable_fc(instance_feature).reshape(bs, num_anchor, self.num_learnable_pts, 3)) - 0.5) # [-0.5,+0.5]
            scale = torch.cat([scale, learnable_scale * self.learnable_fixed_scale], dim=-2) # each gs has 13 sampling pts, 7/13 fixed, 6/13 learnable
        
        t_scales = anchor[..., None, 3:6]
        t_scales = safe_sigmoid(t_scales)
        t_scales = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * t_scales

        key_points = scale * t_scales # this is 13 offset w.r.t each gs mean
        rots = anchor[..., 6:10]
        rotation_mat = get_rotation_matrix(rots).transpose(-1, -2)
        
        key_points = torch.matmul(rotation_mat[:, :, None], key_points[..., None]).squeeze(-1)

        xyz = anchor[..., :3]
        xyz = safe_sigmoid(xyz)
        
        xxx = xyz[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        yyy = xyz[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        zzz = xyz[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
        xyz = torch.stack([xxx, yyy, zzz], dim=-1)
        key_points = key_points + xyz.unsqueeze(2)
        
        return key_points


class Fusion(BaseModule):
    def __init__(
        self,
        embed_dims=128
    ):
        super(Fusion, self).__init__()
        self.gate = nn.Sequential(nn.LayerNorm(embed_dims),
                                  nn.Linear(embed_dims, embed_dims // 2, bias=False),
                                  nn.GELU(),
                                  nn.Linear(embed_dims // 2, 2, bias=True))
        
        self.cam_atten = nn.Sequential(nn.LayerNorm(embed_dims),
                                       nn.Linear(embed_dims, embed_dims // 2, bias=False),
                                       nn.GELU(),
                                       nn.Linear(embed_dims // 2, embed_dims, bias=True),
                                       nn.Sigmoid())
        
        self.lidar_atten = nn.Sequential(nn.LayerNorm(embed_dims),
                                         nn.Linear(embed_dims, embed_dims // 2, bias=False),
                                         nn.GELU(),
                                         nn.Linear(embed_dims // 2, embed_dims, bias=True),
                                         nn.Sigmoid())
        
        self.fusion1_mlp = nn.Sequential(nn.LayerNorm(embed_dims*2),
                                         nn.Linear(embed_dims*2, embed_dims, bias=False),
                                         nn.GELU(),
                                         nn.Linear(embed_dims, embed_dims, bias=True))
        
        self.fusion2_mlp = nn.Sequential(nn.LayerNorm(embed_dims),
                                         nn.Linear(embed_dims, embed_dims, bias=False),
                                         nn.GELU(),
                                         nn.Linear(embed_dims, embed_dims, bias=True))
        
        self.fusion3_mlp = nn.Sequential(nn.LayerNorm(embed_dims*2),
                                         nn.Linear(embed_dims*2, embed_dims, bias=False),
                                         nn.GELU(),
                                         nn.Linear(embed_dims, embed_dims, bias=True))
        
        nn.init.zeros_(self.fusion1_mlp[-1].weight)
        nn.init.zeros_(self.fusion2_mlp[-1].weight)
        nn.init.zeros_(self.fusion3_mlp[-1].weight)
        
        
    def forward(self,
                cam_feat,
                lidar_feat):
        
        joint = 0.5 * (cam_feat + lidar_feat)
        logits = self.gate(joint)
        weights = F.softmax(logits, dim=-1)
        w_cam, w_lidar = weights[..., 0:1], weights[..., 1:2]
        
        cam_gate = self.cam_atten(cam_feat)
        lidar_gate = self.lidar_atten(lidar_feat)
        cam_out = (1.0 + lidar_gate) * cam_feat
        lidar_out = (1.0 + cam_gate) * lidar_feat
        
        fuse_feat_concat = torch.cat([cam_out, lidar_out], dim=-1)
        mid_feat_1 = self.fusion1_mlp(fuse_feat_concat)
        
        fuse_feat_sum = w_cam * cam_feat + w_lidar * lidar_feat
        mid_feat_2 = self.fusion2_mlp(fuse_feat_sum)
        
        mid_feat = torch.cat([mid_feat_1, mid_feat_2], dim=-1)
        final_feat = self.fusion3_mlp(mid_feat) + fuse_feat_sum 
        
        return final_feat
    

class DeformableFeatureAggregation3D(BaseModule):
    def __init__(
        self,
        embed_dims, # 128
        num_head, # number of head
        num_levels, # 3
        num_cams,   # 6
        dbound,
        pol_range,
        grid_size_pol,
        use_occ3d=False,
        use_semanticKITTI=False
    ):
        super(DeformableFeatureAggregation3D, self).__init__()
        if embed_dims % num_head != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, "
                f"but got {embed_dims} and {num_head}"
            )
        self.use_occ3d = use_occ3d
        self.use_semanticKITTI = use_semanticKITTI
        self.grid_size_pol = torch.tensor(grid_size_pol, dtype=torch.float32)
        self.min_bound_pol = torch.tensor(pol_range[:3])
        self.max_bound_pol = torch.tensor(pol_range[3:])
        self.intervals_pol = (self.max_bound_pol - self.min_bound_pol) / self.grid_size_pol
        self.lidar_atten_weights = nn.Sequential(nn.Linear(embed_dims, embed_dims//2),
                                                 nn.LayerNorm(embed_dims//2),
                                                 nn.ReLU(),
                                                 nn.Linear(embed_dims//2, 13))
        self.dbound = dbound
        self.depth_channels = int((self.dbound[1] - self.dbound[0]) // self.dbound[2])
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_head = num_head  # number of head 4
        self.num_cams = num_cams
        self.layer_norm = nn.LayerNorm(embed_dims)
        if self.use_occ3d:
            self.kps_generator = SamplePointsGenerator(embed_dims=embed_dims,
                                                    num_learnable_pts=6,
                                                    learnable_fixed_scale=6.0,
                                                    fix_scale=[[0, 0, 0],
                                                                [0.45, 0, 0],
                                                                [-0.45, 0, 0],
                                                                [0, 0.45, 0],
                                                                [0, -0.45, 0],
                                                                [0, 0, 0.45],
                                                                [0, 0, -0.45]],
                                                    pc_range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
                                                    scale_range=[0.01, 1.8])
        elif self.use_semanticKITTI:
            self.kps_generator = SamplePointsGenerator(embed_dims=embed_dims,
                                                    num_learnable_pts=6,
                                                    learnable_fixed_scale=6.0,
                                                    fix_scale=[[0, 0, 0],
                                                                [0.45, 0, 0],
                                                                [-0.45, 0, 0],
                                                                [0, 0.45, 0],
                                                                [0, -0.45, 0],
                                                                [0, 0, 0.45],
                                                                [0, 0, -0.45]],
                                                    pc_range=[0, -25.6, -2.0, 51.2, 25.6, 4.4],
                                                    scale_range=[0.01, 1.8])
        else:
            self.kps_generator = SamplePointsGenerator(embed_dims=embed_dims,
                                                    num_learnable_pts=6,
                                                    learnable_fixed_scale=6.0,
                                                    fix_scale=[[0, 0, 0],
                                                                [0.45, 0, 0],
                                                                [-0.45, 0, 0],
                                                                [0, 0.45, 0],
                                                                [0, -0.45, 0],
                                                                [0, 0, 0.45],
                                                                [0, 0, -0.45]],
                                                    pc_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0],
                                                    scale_range=[0.01, 1.8])
        self.im2col_step = 64
        self.num_pts = 13
        self.attention_weights = nn.Linear(embed_dims, num_head * num_levels * self.num_pts)
        self.feature_fusion = Fusion(embed_dims)
        
    @staticmethod
    def project_points(key_points, 
                       img_metas,
                       dbound):
        
        lidar2img = []
        image_wh = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
            image_wh.append(img_meta['img_shape'])
        lidar2img = torch.tensor(np.stack(lidar2img)).float().to(key_points.device)
        image_wh = torch.tensor(np.stack(image_wh)).float().to(key_points.device)        

        pts_extend = torch.cat([key_points, torch.ones_like(key_points[..., :1])], dim=-1)
        ref_pts_cam = torch.matmul(lidar2img[:, :, None, None], pts_extend[:, None, ..., None]).squeeze(-1)
        depth = ref_pts_cam[..., 2:3].clone().detach()
        eps = 1e-5
        mask = (depth > eps)
        ref_pts_cam[..., 0:2] = ref_pts_cam[..., 0:2] / torch.maximum(depth, torch.ones_like(depth) * eps) # ud,vd / d -> u,v
        ref_pts_cam[..., 0] /= image_wh[0][0] # normalize into [0, 1]
        ref_pts_cam[..., 1] /= image_wh[0][1] # normalize into [0, 1]
        ref_pts_cam[..., 2]  = (ref_pts_cam[..., 2] - dbound[0]) / (dbound[1]-dbound[0]) # normalize into [0, 1]
        ref_pts_cam = ref_pts_cam[..., :3]
        
        mask = (mask & (ref_pts_cam[..., 2:3] < 1.0)
                    & (ref_pts_cam[..., 2:3] > 0.0) 
                    & (ref_pts_cam[..., 1:2] < 1.0)
                    & (ref_pts_cam[..., 1:2] > 0.0)
                    & (ref_pts_cam[..., 0:1] < 1.0)
                    & (ref_pts_cam[..., 0:1] > 0.0))
        
        mask = torch.nan_to_num(mask).squeeze(-1)
         
        return ref_pts_cam, mask
    
    
    def cross_atten_3d(self, 
                       feat_flatten, 
                       query, 
                       depth_dist, 
                       ref_pts_cam, 
                       mask,
                       spatial_shapes_3D,
                       level_start_index):
        
        slot = torch.zeros_like(query)
        indexes = []
        for i, mask_per_img in enumerate(mask):
            index_query_per_img = mask_per_img.sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])
        
        queries_rebatch = query.new_zeros([self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = ref_pts_cam.new_zeros([self.num_cams, max_len, self.num_pts, 3])

        for i, reference_points_per_img in enumerate(ref_pts_cam):   
            index_query_per_img = indexes[i]
            queries_rebatch[i, :len(index_query_per_img)] = query[index_query_per_img]
            reference_points_rebatch[i, :len(index_query_per_img)] = reference_points_per_img[index_query_per_img]
        
        reference_points_rebatch = reference_points_rebatch[:, :, None, None, :, :].repeat(1, 1, self.num_head, self.num_levels, 1, 1)
        attention_weights = self.attention_weights(queries_rebatch).view(self.num_cams, max_len, self.num_head, self.num_levels * self.num_pts)
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(self.num_cams, 
                                                   max_len,
                                                   self.num_head,
                                                   self.num_levels,
                                                   self.num_pts)
        
        output, depth_score = MultiScale3DDeformableAttnFunction_fp32.apply(feat_flatten, 
                                                                            depth_dist, 
                                                                            spatial_shapes_3D, 
                                                                            level_start_index, 
                                                                            reference_points_rebatch, 
                                                                            attention_weights, 
                                                                            self.im2col_step)
        for i, index_query_per_img in enumerate(indexes):
            slot[index_query_per_img] += output[i, :len(index_query_per_img)]
        
        count = mask.sum(-1) > 0
        count = count.permute(1, 0).sum(-1) # 25600, 6 -> 25600
        count = torch.clamp(count, min=1.0)
        slot = slot / count[..., None]
        
        return slot[None]
    
    def cart_real2pol_vox(self, input_xyz):
        rho = torch.sqrt(input_xyz[..., 0] ** 2 + input_xyz[..., 1] ** 2)
        phi = torch.arctan2(input_xyz[..., 1], input_xyz[..., 0])
        output_rdz =  torch.stack((rho, phi, input_xyz[..., 2]), axis=-1)
        # output_rdz = output_rdz.clamp(min=self.min_bound_pol.to(output_rdz.device), max=self.max_bound_pol.to(output_rdz.device))
        indices = (output_rdz - self.min_bound_pol.to(output_rdz.device)) / self.intervals_pol.to(output_rdz.device)
        # indices = indices.to(torch.int32)
        
        return indices
    
    def sample_lidar_feat(self,
                          vol_lidar,
                          key_points,
                          query):
        key_points = self.cart_real2pol_vox(key_points)
        features = vol_lidar.features
        W, H, D = (int(size) for size in vol_lidar.spatial_shape)
        B, Q, K, _ = key_points.shape
        C = features.shape[-1]

        if B != vol_lidar.batch_size:
            raise ValueError(
                f"LiDAR batch size mismatch: key points have {B} batches, "
                f"but the sparse volume has {vol_lidar.batch_size}"
            )

        # Linear keys let us find active sparse voxels without materializing the
        # dense [B, C, W, H, D] feature volume. The same volume is sampled by
        # every transformer layer, so keep its index-only lookup on the volume.
        sparse_lookup = getattr(vol_lidar, "_tfusion_sampling_lookup", None)
        if sparse_lookup is None:
            indices = vol_lidar.indices.long()
            sparse_keys = (((indices[:, 0] * W + indices[:, 1]) * H
                            + indices[:, 2]) * D + indices[:, 3])
            sparse_keys, sort_order = sparse_keys.sort()
            sparse_lookup = (indices, sparse_keys, sort_order)
            vol_lidar._tfusion_sampling_lookup = sparse_lookup
        indices, sparse_keys, sort_order = sparse_lookup

        point_weights = safe_sigmoid(self.lidar_atten_weights(query))
        if features.shape[0] == 0:
            zero_feature = features.sum(dim=0).view(1, 1, C)
            return point_weights.sum(-1, keepdim=True) * zero_feature

        corner_offsets = indices.new_tensor([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ])
        spatial_shape = indices.new_tensor([W, H, D])

        # Bound each matched-feature gather to about 128 MiB. Attention over K
        # is reduced per chunk, so [B, Q, K, C] is never materialized in full.
        max_gather_bytes = 128 * 1024 * 1024
        bytes_per_sample = corner_offsets.shape[0] * C * features.element_size()
        max_sample_points = min(
            65536, max(1, max_gather_bytes // bytes_per_sample)
        )
        queries_per_chunk = max(1, max_sample_points // (B * K))
        output_chunks = []

        for start in range(0, Q, queries_per_chunk):
            end = min(start + queries_per_chunk, Q)
            chunk_size = end - start
            sample_points = key_points[:, start:end].reshape(-1, 3)

            # Reproduce grid_sample's align_corners=False coordinate transform,
            # including its float rounding at exact voxel boundaries.
            sample_shape = sample_points.new_tensor([W, H, D])
            normalized_points = sample_points / sample_shape * 2 - 1
            source_points = ((normalized_points + 1) * sample_shape - 1) / 2
            lower = source_points.floor().long()
            fraction = source_points - lower.type_as(source_points)

            neighbors = lower[:, None, :] + corner_offsets[None, :, :]
            valid = ((neighbors >= 0)
                     & (neighbors < spatial_shape[None, None, :])).all(-1)

            batch_indices = torch.arange(
                B, device=indices.device, dtype=indices.dtype
            )[:, None, None]
            batch_indices = batch_indices.expand(B, chunk_size, K).reshape(-1)
            neighbor_keys = (((batch_indices[:, None] * W
                               + neighbors[..., 0]) * H
                              + neighbors[..., 1]) * D
                             + neighbors[..., 2])

            positions = torch.searchsorted(sparse_keys, neighbor_keys)
            positions_in_range = positions < sparse_keys.numel()
            positions = positions.clamp_max(sparse_keys.numel() - 1)
            matches = (valid & positions_in_range
                       & (sparse_keys[positions] == neighbor_keys))

            corner_fractions = torch.where(
                corner_offsets[None].bool(),
                fraction[:, None, :],
                1 - fraction[:, None, :],
            )
            interpolation_weights = corner_fractions.prod(-1)

            matched = matches.nonzero(as_tuple=False)
            sample_indices = matched[:, 0]
            corner_indices = matched[:, 1]
            feature_indices = sort_order[
                positions[sample_indices, corner_indices]
            ]
            matched_weights = interpolation_weights[
                sample_indices, corner_indices
            ].type_as(features)
            contributions = (
                features[feature_indices] * matched_weights[:, None]
            )

            sampled = features.new_zeros(sample_points.shape[0], C)
            sampled = sampled.index_add(0, sample_indices, contributions)
            sampled = sampled.view(B, chunk_size, K, C)
            chunk_weights = point_weights[:, start:end]
            output_chunks.append(
                (chunk_weights[..., None] * sampled).sum(-2)
            )

        return torch.cat(output_chunks, dim=1)
    
    
    def forward(self,
                vol_lidar,
                anchor,
                instance_feature,
                anchor_embed,
                feat_flatten,
                spatial_shapes_3D,
                level_start_index,
                depth_dist,
                metas):
        
        identity = instance_feature
        query = instance_feature + anchor_embed # lidar_feats
        key_points = self.kps_generator(anchor, query) # instance_feature
        lidar_outs = self.sample_lidar_feat(vol_lidar, key_points, query) # key_points, new: anchor[..., :3]
        ref_pts_cams, masks = self.project_points(key_points, metas, self.dbound)        
        vis_outs = multi_apply(self.cross_atten_3d, 
                                feat_flatten, 
                                query, 
                                depth_dist, 
                                ref_pts_cams, 
                                masks,
                                spatial_shapes_3D,
                                level_start_index)[0]
        vis_outs = torch.stack(vis_outs)
        fused_feat = self.feature_fusion(vis_outs, lidar_outs)
        res = self.layer_norm(fused_feat + identity) 
        
        return res

    

    

    

    
