import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import numpy as np
from mmengine.runner.amp import autocast
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from mmdet.models.utils.misc import multi_apply
from spconv.pytorch import SparseConvTensor, SparseMaxPool3d
from lib.pointops.functions.pointops import furthestsampling as farthest_point_sampling 
from .safe_ops import safe_sigmoid, safe_inverse_sigmoid
from .sampler import DistributionSampler

@MODELS.register_module()
class Cylindrical_Partition(BaseModule):
    def __init__(self, 
                 num_lidar_anchor,
                 num_cam_anchor,
                 k,
                 grid_size_pol, 
                 grid_size_occ,
                 in_channels=11, 
                 out_channels=128,
                 dbound=[],
                 pc_range=[],
                 pol_range=[],
                 lidar_encoder=None):
        super().__init__()
        self.num_lidar_anchor = num_lidar_anchor
        self.num_cam_anchor = num_cam_anchor
        self.empty_label = 0
        self.TopK = k
        self.dbound = dbound
        self.lidar_encoder = MODELS.build(lidar_encoder)
        self.sampler = DistributionSampler()
        self.num_samples = int((self.dbound[1] - self.dbound[0]) // self.dbound[2])
        self.grid_size_pol = torch.tensor(grid_size_pol)
        self.grid_size_occ = torch.tensor(grid_size_occ)
        self.pc_range = torch.tensor(pc_range)
        self.min_bound = torch.tensor(pc_range[:3])
        self.min_bound_pol = torch.tensor(pol_range[:3])
        self.max_bound = torch.tensor(pc_range[3:])
        self.max_bound_pol = torch.tensor(pol_range[3:])
        self.intervals = (self.max_bound - self.min_bound) / self.grid_size_occ
        self.intervals_pol = (self.max_bound_pol - self.min_bound_pol) / self.grid_size_pol
        self.point_mlp = nn.Sequential(nn.BatchNorm1d(in_channels),
                                       nn.Linear(in_channels, 64),
                                       nn.BatchNorm1d(64),
                                       nn.ReLU(),
                                       nn.Linear(64, 128),
                                       nn.BatchNorm1d(128),
                                       nn.ReLU(),
                                       nn.Linear(128, 256),
                                       nn.BatchNorm1d(256),
                                       nn.ReLU(),
                                       nn.Linear(256, out_channels))
        self.fusion_mlp = nn.Sequential(nn.LayerNorm(out_channels*2),
                                        nn.Linear(out_channels*2, out_channels, bias=False),
                                        nn.GELU(),
                                        nn.Linear(out_channels, out_channels, bias=True))
        self.attention_weights = nn.Sequential(nn.Linear(out_channels, out_channels//2),
                                               nn.LayerNorm(out_channels//2),
                                               nn.ReLU(),
                                               nn.Linear(out_channels//2, 3))
        self.projection = nn.Linear(out_channels*3, self.num_samples + 1)
        self.depth_bins = torch.arange(self.dbound[0] + self.dbound[2] / 2,
                                       self.dbound[1],
                                       self.dbound[2])
        
    
    # Check which rows in b are in a
    def rows_in(self, a: torch.Tensor, b: torch.Tensor):
        assert a.ndim == 2 and b.ndim == 2 and a.shape[1] == 3 and b.shape[1] == 3, "Only supports shape (N, 3)"
        a = a.long()
        b = b.long()
        device = a.device
        assert b.device == device, "a and b must be on the same device"
        
        # ---- handle empty cases ----
        if b.numel() == 0:  # b has shape (0, 3)
            return torch.empty((b.shape[0],), device=b.device, dtype=torch.bool)
        if a.numel() == 0:  # a has shape (0, 3), b non-empty
            return torch.zeros((b.shape[0],), device=b.device, dtype=torch.bool)
        
        SHIFT = 21
        MASK  = (1 << SHIFT) - 1  # 0..2^21 - 1
        
        a_min = a.min(dim=0).values
        b_min = b.min(dim=0).values
        bias = torch.minimum(a_min, b_min)  # keep everything non-negative after biasing

        a_biased = a - bias
        b_biased = b - bias

        def pack_3cols(x):
            # (x0 & MASK) << (2*SHIFT) | (x1 & MASK) << SHIFT | (x2 & MASK)
            x0 = (x[:, 0] & MASK).to(torch.int64)
            x1 = (x[:, 1] & MASK).to(torch.int64)
            x2 = (x[:, 2] & MASK).to(torch.int64)
            return (x0 << (2 * SHIFT)) | (x1 << SHIFT) | x2

        a_ids = pack_3cols(a_biased)
        b_ids = pack_3cols(b_biased)

        a_ids_unique = torch.unique(a_ids, sorted=True)

        pos = torch.searchsorted(a_ids_unique, b_ids, right=False)
        in_bounds = pos < a_ids_unique.numel()
        mask = torch.zeros_like(in_bounds, dtype=torch.bool)
        mask[in_bounds] = (a_ids_unique[pos[in_bounds]] == b_ids[in_bounds])

        return mask


    def points_within_radius(self, A, B, r, chunk_size=1024):
        N = A.shape[0]
        device = A.device
        result_mask = torch.zeros(N, dtype=torch.bool, device=device)

        r2 = r * r  # Compare squared distances to avoid sqrt

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            a_chunk = A[start:end]  # (chunk, 3)

            # Compute squared distances without expanding to full (chunk, M, 3)
            # (chunk, 1, 3) - (1, M, 3) => (chunk, M, 3)
            dist2 = ((a_chunk[:, None, :] - B[None, :, :]) ** 2).sum(dim=2)

            # Check if any point in B is within radius
            close = (dist2 <= r2).any(dim=1)
            result_mask[start:end] = close

        return result_mask
    
        
    def init_random_anchors(self, num_anchor, embed_dim, type):
        xyz = torch.rand(num_anchor, 3, dtype=torch.float)
        anchor = nn.Parameter(xyz, True)
        if type == 'lidar':
            lidar_feats = torch.zeros(num_anchor, embed_dim) # empty lidar feats
            return anchor, lidar_feats
        elif type == 'cam':
            return anchor

    def cat_logit2polar_vox(self, input_logit):
        cart_inter = safe_sigmoid(input_logit)
        cart_real = cart_inter * (self.max_bound.to(input_logit.device) - self.min_bound.to(input_logit.device)) + self.min_bound.to(input_logit.device)
        pol_vox = self.cart_real2pol_vox(cart_real)
        
        return pol_vox
    
    def cart_real2pol_vox(self, input_xyz):
        rho = torch.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
        phi = torch.arctan2(input_xyz[:, 1], input_xyz[:, 0])
        output_rdz =  torch.stack((rho, phi, input_xyz[:, 2]), axis=1)
        output_rdz = output_rdz.clamp(min=self.min_bound_pol.to(output_rdz.device), max=self.max_bound_pol.to(output_rdz.device))
        indices = (output_rdz - self.min_bound_pol.to(output_rdz.device)) / self.intervals_pol.to(output_rdz.device)
        indices = indices.to(torch.int32)
        
        return indices

    def polar_vox2cat_real(self, grid_ind):
        xyz_polar = (grid_ind + 0.5) * self.intervals_pol.to(grid_ind.device) + self.min_bound_pol.to(grid_ind.device)
        r = xyz_polar[:, 0]
        theta = xyz_polar[:, 1]
        z = xyz_polar[:, 2]
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        xyz_cat = torch.stack([x, y, z], dim=1)
        
        return xyz_cat
    
    def polar_vox2cat_inter(self, grid_ind):
        xyz_polar = (grid_ind + 0.5) * self.intervals_pol.to(grid_ind.device) + self.min_bound_pol.to(grid_ind.device)
        r = xyz_polar[:, 0]
        theta = xyz_polar[:, 1]
        z = xyz_polar[:, 2]
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        xyz_cat_real = torch.stack([x, y, z], dim=1)
        xyz = (xyz_cat_real - self.min_bound.to(grid_ind.device)) / (self.max_bound.to(grid_ind.device) - self.min_bound.to(grid_ind.device))

        return xyz 
    
    def sample_img_feat(self, lidar_feat, img_shape, vis_feats, samp_loc, lidar2img):
        eps = 1e-5
        num_query = samp_loc.shape[0]
        num_cam = lidar2img.shape[0]
        samp_loc = torch.cat([samp_loc, torch.ones([samp_loc.shape[0],1]).to(samp_loc.device)], dim=1)
        lidar2img = lidar2img.view(num_cam, 1, 4, 4).repeat(1, num_query, 1, 1)
        samp_loc = samp_loc.view(1, num_query, 4).repeat(num_cam, 1, 1).unsqueeze(-1)
        pt_cam = torch.matmul(lidar2img, samp_loc).squeeze(-1)
        z = pt_cam[..., 2:3] # Z coordinate
        valid_mask = (z > eps) & torch.isfinite(z)
        pt_cam = pt_cam[..., 0:2] / z.clamp_min(eps)  # prevent zero-division, convert to (u,v) coordinate
        (w, h) = img_shape # 1600, 928
        pt_cam[..., 0] /= w # map to [0,1)
        pt_cam[..., 1] /= h # map to [0,1)

        # grid_sample returns zeros for finite out-of-bounds coordinates, but it
        # does not reject points behind the camera and cannot safely handle
        # NaN/Inf coordinates. Track visibility explicitly before sampling.
        valid_mask = (
            valid_mask
            & torch.isfinite(pt_cam).all(dim=-1, keepdim=True)
            & (pt_cam[..., 0:1] >= 0.0)
            & (pt_cam[..., 0:1] < 1.0)
            & (pt_cam[..., 1:2] >= 0.0)
            & (pt_cam[..., 1:2] < 1.0)
        )
        pt_cam = (pt_cam - 0.5) * 2  # [0,1] to [-1,1] to do grid_sample
        pt_cam = torch.where(valid_mask.expand_as(pt_cam), pt_cam, torch.full_like(pt_cam, 2.0))
        pt_cam = pt_cam.view(num_cam, num_query, 1, 2)
        valid_mask = valid_mask.view(num_cam, num_query, 1)
        feat_sampled = []
        for feat in vis_feats:
            sampled_feat = F.grid_sample(feat.float(), pt_cam, align_corners=False).squeeze(-1)
            sampled_feat = sampled_feat.permute(0, 2, 1)
            sampled_feat = (sampled_feat * valid_mask.to(sampled_feat.dtype)).sum(0)
            feat_sampled.append(sampled_feat)
        feat_sampled = torch.stack(feat_sampled, dim=1)
        atten_weight = self.attention_weights(lidar_feat)
        atten_weight = atten_weight.softmax(dim=-1)
        final_feat = atten_weight[...,None] * feat_sampled
        final_feat = final_feat.sum(1)
        
        return final_feat
    
    
    def process_lidar_data(self, 
                           lidar_pts, 
                           lidar_grid_ind,
                           img_feats,
                           img_metas):
        
        device = lidar_pts[0].get_device()
        ########## Process Lidar data ##############
        cat_lidar_pt_ind = []
        for i_batch, res in enumerate(lidar_grid_ind):
            cat_lidar_pt_ind.append(F.pad(lidar_grid_ind[i_batch], (1, 0), 'constant', value=i_batch))
        
        cat_lidar_pt_fea = torch.cat(lidar_pts, dim=0)
        cat_lidar_pt_ind = torch.cat(cat_lidar_pt_ind, dim=0)
        lidar_pt_num = cat_lidar_pt_ind.shape[0]
        # shuffle the data
        shuffled_ind_lidar = torch.randperm(lidar_pt_num, device=device)
        cat_lidar_pt_fea = cat_lidar_pt_fea[shuffled_ind_lidar, :]
        cat_lidar_pt_ind = cat_lidar_pt_ind[shuffled_ind_lidar, :]
        # unique xy grid index
        unq_lidar, unq_inv_lidar, unq_cnt_lidar = torch.unique(cat_lidar_pt_ind, return_inverse=True, return_counts=True, dim=0)
        # process feature
        processed_cat_pt_fea_lidar = self.point_mlp(cat_lidar_pt_fea)
        # mean pooling
        pooled_data_lidar = torch_scatter.scatter_mean(processed_cat_pt_fea_lidar, unq_inv_lidar, dim=0)
        coors_lidar = unq_lidar.int()
        batch_size = coors_lidar[-1][0] + 1
        ##### Aggregate Visual-Feature ##########
        for bs in range(batch_size):
            idx = (coors_lidar[:,0] == bs)
            cur_bs_polcoors_lidar = coors_lidar[idx][..., 1:]
            cur_bs_pooled_lidar = pooled_data_lidar[idx]
            ####### Aggregate semantic info for lidar ########
            lidar2img = img_metas[bs]['lidar2img']
            lidar2img = torch.tensor(lidar2img,device=device,dtype=torch.float32)
            cur_bs_cartcoor_lidar = self.polar_vox2cat_real(cur_bs_polcoors_lidar) #cart_coords[idx][:, 1:]
            cur_bs_vis_feat = [vis_feat[bs] for vis_feat in img_feats] #img_feats[bs]
            cam_feat = self.sample_img_feat(cur_bs_pooled_lidar,
                                            img_metas[bs]['img_shape'], 
                                            cur_bs_vis_feat, 
                                            cur_bs_cartcoor_lidar, 
                                            lidar2img)
            fused_feat = torch.cat([cur_bs_pooled_lidar, cam_feat], dim=1)
            fused_feat = self.fusion_mlp(fused_feat)
            pooled_data_lidar[idx] = fused_feat
        ######################################################
        #### Cylinder3D 3D Backbone for Lidar-Feature ########
        vol_lidar = self.lidar_encoder(pooled_data_lidar, coors_lidar, batch_size)
        ######################################################
        out_coor = []
        for bs in range(batch_size):
            idx = (coors_lidar[:,0] == bs)
            cur_bs_polcoors_lidar = coors_lidar[idx][..., 1:]

            if cur_bs_polcoors_lidar.shape[0] >= self.num_lidar_anchor:
                scanidx = farthest_point_sampling(self.polar_vox2cat_real(cur_bs_polcoors_lidar).contiguous(), 
                                                  torch.tensor([cur_bs_polcoors_lidar.shape[0]], device=device, dtype=torch.int),
                                                  torch.tensor([self.num_lidar_anchor], device=device, dtype=torch.int))
                out_coor.append(safe_inverse_sigmoid(self.polar_vox2cat_inter(cur_bs_polcoors_lidar[scanidx]))) 
            else:
                out_coor.append(safe_inverse_sigmoid(self.polar_vox2cat_inter(cur_bs_polcoors_lidar)))

        
        
        return out_coor, vol_lidar 
    
    @torch.no_grad()
    def Gen_Cam_pts(self, merged_feats, img_metas):
        b, n, _, h, w = merged_feats.shape
        device = merged_feats.device
        dtype  = merged_feats.dtype

        # ---------- 1) calibration tensors on GPU, no numpy round-trip ----------
        lidar2img = torch.stack(
            [torch.as_tensor(m['lidar2img'], device=device, dtype=dtype)
            for m in img_metas], dim=0
        )  # expected shape: [b, n, 4, 4] or [b, 6, 4, 4] etc.

        image_wh = torch.stack(
            [torch.as_tensor(m['img_shape'], device=device, dtype=dtype)
            for m in img_metas], dim=0
        )  # typically [b, n, 2] or [b, 2]

        # keep the original semantics of your tile if needed
        if image_wh.ndim == 2:
            # original code: image_wh = image_wh[:, None].tile(1, 6, 1)
            # using expand avoids an extra allocation
            image_wh = image_wh[:, None].expand(-1, n, -1)

        img2lidar = torch.inverse(lidar2img)          # [b, n, 4, 4]

        # split into rotation and translation; last row is [0,0,0,1] in usual setups
        R = img2lidar[..., :3, :3]                    # [b, n, 3, 3]
        t = img2lidar[..., :3, 3]                     # [b, n, 3]

        # ---------- 2) pixel centers (u, v) in *normalized* coords ----------
        u = (torch.arange(w, dtype=dtype, device=device) + 0.5) / w  # [w]
        v = (torch.arange(h, dtype=dtype, device=device) + 0.5) / h  # [h]

        v_grid, u_grid = torch.meshgrid(v, u, indexing='ij')         # [h, w], [h, w]
        uv = torch.stack([u_grid, v_grid], dim=-1)                   # [h, w, 2]

        # scale to pixel coordinates and broadcast to [b, n, h, w, 2]
        uv = uv.view(1, 1, h, w, 2) * image_wh.view(b, n, 1, 1, 2)   # [b, n, h, w, 2]

        # ---------- 3) depths ----------
        D = self.num_samples
        depths = self.depth_bins.to(device=device, dtype=dtype).view(1, 1, 1, 1, D, 1)
        # shape: [1, 1, 1, 1, D, 1]

        # ---------- 4) build anchor points *directly* without uvd / uvd1 ----------
        # uv[..., 0] / uv[..., 1] are pixel coords; add depth dimension
        u_pix = uv[..., 0:1].unsqueeze(-1)     # [b, n, h, w, 1, 1]
        v_pix = uv[..., 1:2].unsqueeze(-1)     # [b, n, h, w, 1, 1]

        # start from translation t, broadcasted to full shape, and clone to own storage
        anchor = t.view(b, n, 1, 1, 1, 3).expand(b, n, h, w, D, 3).clone()

        # columns of R as 3D vectors
        r0 = R[..., :, 0].view(b, n, 1, 1, 1, 3)  # column 0
        r1 = R[..., :, 1].view(b, n, 1, 1, 1, 3)  # column 1
        r2 = R[..., :, 2].view(b, n, 1, 1, 1, 3)  # column 2

        # X_lidar = t + (u*d)*r0 + (v*d)*r1 + d*r2
        ud = u_pix * depths     # [b, n, h, w, D, 1]
        vd = v_pix * depths     # [b, n, h, w, D, 1]

        anchor += r0 * ud       # uses broadcasting
        anchor += r1 * vd
        anchor += r2 * depths   # depths is already [*, D, 1]

        # anchor: [b, n, h, w, D, 3]
        return anchor
    
        
    def filter_cam_merge_lidar(self, lidar_coor_logit, sampled_anchor, disable_mask):
        device = lidar_coor_logit.device
        lidar_pol_vox = self.cat_logit2polar_vox(lidar_coor_logit)
        cur_sampled_anchor = sampled_anchor[~disable_mask] # get non-empty cam-ray pts
        cur_oob_mask = (cur_sampled_anchor[..., 0] < self.pc_range[0]) | (cur_sampled_anchor[..., 0] >= self.pc_range[3]) | \
                        (cur_sampled_anchor[..., 1] < self.pc_range[1]) | (cur_sampled_anchor[..., 1] >= self.pc_range[4]) | \
                        (cur_sampled_anchor[..., 2] < self.pc_range[2]) | (cur_sampled_anchor[..., 2] >= self.pc_range[5])
        cur_inlier_cart_real = cur_sampled_anchor[~cur_oob_mask]
        cur_inlier_pol_vox = self.cart_real2pol_vox(cur_inlier_cart_real)
        unq_cam_pol_vox, unq_inv_cam, unq_cnt_cam = torch.unique(cur_inlier_pol_vox, return_inverse=True, return_counts=True, dim=0)
        common_idx = self.rows_in(lidar_pol_vox, unq_cam_pol_vox) 
        filtered_cam_pol_vox = unq_cam_pol_vox[~common_idx]
        idx = self.points_within_radius(self.polar_vox2cat_real(filtered_cam_pol_vox), self.polar_vox2cat_real(lidar_pol_vox), r=5.0)
        filtered_cam_pol_vox = filtered_cam_pol_vox[idx]
        
        if filtered_cam_pol_vox.shape[0] >= (self.num_cam_anchor + self.num_lidar_anchor - lidar_coor_logit.shape[0]):
            scanidx = farthest_point_sampling(self.polar_vox2cat_real(filtered_cam_pol_vox).contiguous(), 
                                                torch.tensor([filtered_cam_pol_vox.shape[0]], device=device, dtype=torch.int),
                                                torch.tensor([self.num_cam_anchor + self.num_lidar_anchor - lidar_coor_logit.shape[0]], device=device, dtype=torch.int))
            final_coor = torch.cat([lidar_coor_logit, 
                                    safe_inverse_sigmoid(self.polar_vox2cat_inter(filtered_cam_pol_vox[scanidx]))], 
                                    dim=0)
            
        else:
            anchor_coor_rand = self.init_random_anchors(self.num_cam_anchor + self.num_lidar_anchor - lidar_coor_logit.shape[0] - filtered_cam_pol_vox.shape[0], 
                                                        None, 
                                                        type='cam')
            final_coor = torch.cat([lidar_coor_logit,
                                    safe_inverse_sigmoid(self.polar_vox2cat_inter(filtered_cam_pol_vox)),
                                    safe_inverse_sigmoid(anchor_coor_rand).to(device)], # -> turns into logits
                                    dim=0)
            
        
        return final_coor[None]
    
    def process_cam_data(self, 
                         merged_feats, 
                         img_metas, 
                         lidar_coor_logits):
        
        anchor_pts = self.Gen_Cam_pts(merged_feats, img_metas) # consume 1.5GB GPU memory
        logits = self.projection(merged_feats.permute(0,1,3,4,2))
        dist_prob = safe_sigmoid(logits)
        val, idx = torch.topk(dist_prob, k=self.TopK, dim=-1)
        disable_mask = (dist_prob.argmax(dim=-1, keepdim=True) == self.num_samples).expand(-1, -1, -1, -1, self.TopK)
        sampled_anchor = self.sampler.gather(idx.clamp(max=(self.num_samples-1)), anchor_pts) # b, n, h, w, a, 3
        out_coor = multi_apply(self.filter_cam_merge_lidar,
                                lidar_coor_logits,
                                sampled_anchor,
                                disable_mask)[0]
        
        out_coor = torch.stack(out_coor, dim=0)
        
        return out_coor, logits, anchor_pts     
    
        
        
    def forward(self, merged_feats, img_feats, img_metas, lidar_pts, lidar_grid_ind):
        
        out_coor_lidar, vol_lidar = self.process_lidar_data(lidar_pts, 
                                                            lidar_grid_ind,
                                                            img_feats,
                                                            img_metas)
        
        out_coor, cam_logits, cam_pts = self.process_cam_data(merged_feats, 
                                                              img_metas, 
                                                              out_coor_lidar)
        
        
        return out_coor, vol_lidar, cam_logits, cam_pts 