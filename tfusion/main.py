from mmengine.runner.amp import autocast
from mmdet3d.models import Base3DSegmentor
from mmdet3d.registry import MODELS
from mmdet3d.structures import PointData
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .grid_mask import GridMask
from .loss import CE_wo_softmax, cam_init_loss
from .lovasz_softmax import lovasz_softmax

@MODELS.register_module()
class TFusionOcc(Base3DSegmentor):
    def __init__(self,
                 mul_sup=False,
                 use_occ3d=False,
                 use_semanticKITTI=False,
                 use_grid_mask=False,
                 data_preprocessor=None,
                 backbone=None,
                 second_neck=None,
                 neck=None,
                 cylindrical_partition=None,
                 transformer=None):
        super().__init__(data_preprocessor=data_preprocessor)
        self.mul_sup = mul_sup
        self.occ3d = use_occ3d
        self.semanticKITTI = use_semanticKITTI
        self.use_grid_mask=use_grid_mask
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=1.0)
        self.img_backbone = MODELS.build(backbone)
        self.second_neck = MODELS.build(second_neck)
        self.img_neck = MODELS.build(neck)
        self.cylindrical_partition = MODELS.build(cylindrical_partition)
        self.transformer = MODELS.build(transformer)
        if self.occ3d:
            self.class_weights=torch.tensor([0.5 , 1.06897009, 1.30013094, 1.07253735, 0.94637502, 1.10087012,
                                            1.26960524, 1.06258364, 1.189019,   1.06217292, 1.00595144, 0.85706115,
                                            1.03923299, 0.90867526, 0.8936431,  0.85486129, 0.8527829, 1.01552756])
            self.class_weights = 18 * F.normalize(self.class_weights, 1, -1)
        elif self.semanticKITTI:
            self.class_weights=None
        else:
            self.class_weights=torch.tensor([0.5 , 1.06897009, 1.30013094, 1.07253735, 0.94637502, 1.10087012,
                                            1.26960524, 1.06258364, 1.189019,   1.06217292, 1.00595144, 0.85706115,
                                            1.03923299, 0.90867526, 0.8936431,  0.85486129, 0.8527829])
            self.class_weights = 17 * F.normalize(self.class_weights, 1, -1)

    def multiscale_supervision(self, gt_occ, ratio, gt_shape):
        gt = torch.zeros([gt_shape[0], gt_shape[1], gt_shape[2], gt_shape[3]]).to(gt_occ[0].device).type(torch.long) 
        for i in range(gt.shape[0]):
            coords_x = gt_occ[i][:, 0].to(torch.float) // ratio[0]
            coords_y = gt_occ[i][:, 1].to(torch.float) // ratio[1]
            coords_z = gt_occ[i][:, 2].to(torch.float) // ratio[2]
            coords_x = coords_x.to(torch.long)
            coords_y = coords_y.to(torch.long)
            coords_z = coords_z.to(torch.long)
            coords = torch.stack([coords_x,coords_y,coords_z],dim=1)
            gt[i, coords[:, 0], coords[:, 1], coords[:, 2]] =  gt_occ[i][:, 3]
        
        return gt
    
    # @autocast('cuda',torch.float16)
    def extract_feat(self, img):
        """Extract features of images."""
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
        if self.use_grid_mask:
            img = self.grid_mask(img)  # mask out some grids
        img_feats = self.img_backbone(img)
        merged_feats = self.second_neck(img_feats)
        merged_feats = torch.stack(merged_feats)
        merged_feats = merged_feats.view(B, N, -1, H//8, W//8)
        img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            _, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, N, C, H, W))
        return img_feats_reshaped, merged_feats    
    
    def _forward(self, batch_inputs, batch_data_samples): 
        """Forward training function."""
        imgs = batch_inputs['imgs']
        img_metas = []
        for data_sample in batch_data_samples:
            if not self.occ3d:
                img_meta = dict(lidar2img=data_sample.lidar2img,
                                img_shape=torch.Tensor([imgs.shape[-1],imgs.shape[-2]]))
            else:
                img_meta = dict(lidar2img=data_sample.ego2img,
                                img_shape=torch.Tensor([imgs.shape[-1],imgs.shape[-2]]))
                
            img_metas.append(img_meta)
        
        img_feats, merged_feats = self.extract_feat(imgs)
        anchor_coor, vol_lidar, cam_logits, cam_pts = self.cylindrical_partition(merged_feats.float(),
                                                                                img_feats,
                                                                                img_metas,
                                                                                batch_inputs['pts_pol_feat'],
                                                                                batch_inputs['pts_pol_vox_coords'])
        output = self.transformer(vol_lidar,
                                  batch_inputs['sample_loc'],
                                  anchor_coor,
                                  cam_logits,
                                  img_feats,
                                  batch_inputs['depth_gt'],
                                  img_metas)

        return output, cam_logits, cam_pts # during training return tuple, during testing return single variable
             
    def loss(self,batch_inputs, batch_data_samples):
        out_res, cam_logits, cam_pts = self._forward(batch_inputs,batch_data_samples) # depths
        if self.occ3d:
            GT_lvl0_ori = self.multiscale_supervision(batch_inputs['occ_3d_labels'],[1,1,1],np.array([len(batch_data_samples),200,200,16],dtype=np.int32))
            mask = self.multiscale_supervision(batch_inputs['occ_3d_mask'],[1,1,1],np.array([len(batch_data_samples),200,200,16],dtype=np.int32))
            GT_lvl0 = GT_lvl0_ori * mask
            GT_lvl0[GT_lvl0==0]=255
            GT_lvl0[GT_lvl0==17]=0
            GT_lvl0[GT_lvl0==18]=17
        elif self.semanticKITTI:
            GT_lvl0 = self.multiscale_supervision(batch_inputs['occ_semantickitti_masked'],[1,1,1],np.array([len(batch_data_samples),256,256,32],dtype=np.int32))
        else:
            GT_lvl0 = self.multiscale_supervision(batch_inputs['dense_occ_200'],[1,1,1],np.array([len(batch_data_samples),200,200,16],dtype=np.int32))
        
        loss = dict()
        if not self.mul_sup:
            B,N,Cls = out_res.shape
            loss['loss_voxel_ce'] = 10.0 * CE_wo_softmax(out_res.transpose(1, 2), GT_lvl0.reshape(B,-1), self.class_weights, ignore_index=255) # ignore noisy label 255, .type_as(out_res)
            loss['loss_voxel_lovasz'] = 1.0 * lovasz_softmax(out_res.flatten(0, 1), GT_lvl0.flatten(), ignore=0)  # ignore empty label 0 and noise label 255
        else:
            for idx, out_logit in enumerate(out_res):
                B,N,Cls = out_logit.shape
                loss[f'loss_{idx}_voxel_ce'] = 10.0 * CE_wo_softmax(out_logit.transpose(1, 2), GT_lvl0.reshape(B,-1), self.class_weights.type_as(out_logit), ignore_index=255) * 0.25 * (idx+1)
    
            for idx, out_logit in enumerate(out_res):
                B,N,Cls = out_logit.shape
                loss[f'loss_{idx}_voxel_lovasz'] = 1.0 * lovasz_softmax(out_logit.flatten(0, 1), GT_lvl0.flatten(), ignore=0) * 0.25 * (idx+1)

            
        if self.occ3d:
            vis_gt = GT_lvl0.clone()
            vis_gt[vis_gt==255]=0
            loss['loss_cam_init'] = 1.0 * cam_init_loss(cam_pts, 
                                                        cam_logits, 
                                                        vis_gt, 
                                                        [-40, -40, -1, 40, 40, 5.4], 
                                                        0.4)
        elif self.semanticKITTI:
            vis_gt = GT_lvl0.clone()
            vis_gt[vis_gt==255]=0
            loss['loss_cam_init'] = 1.0 * cam_init_loss(cam_pts, 
                                                        cam_logits, 
                                                        vis_gt, 
                                                        [0, -25.6, -2.0, 51.2, 25.6, 4.4], 
                                                        0.2)
        else:
            loss['loss_cam_init'] = 1.0 * cam_init_loss(cam_pts, 
                                                        cam_logits, 
                                                        GT_lvl0, 
                                                        [-50, -50, -5, 50, 50, 3], 
                                                        0.5)
        return loss
    
    def predict(self, batch_inputs, batch_data_samples):
        """Forward predict function."""
        occ_ori_logits, _, _ = self._forward(batch_inputs, batch_data_samples)
        B,N,Cls = occ_ori_logits.shape
        if self.occ3d:
            lvl0_labels = self.multiscale_supervision(batch_inputs['occ_3d_labels'],[1,1,1],np.array([len(batch_data_samples),200,200,16],dtype=np.int32))
            lvl0_masks = self.multiscale_supervision(batch_inputs['occ_3d_mask'],[1,1,1],np.array([len(batch_data_samples),200,200,16],dtype=np.int32))
            voxels_lvl0 = lvl0_labels.reshape(len(batch_data_samples),-1)
            lvl0_masks = lvl0_masks.reshape(len(batch_data_samples),-1).to(bool)
        elif self.semanticKITTI:
            voxels_lvl0 = self.multiscale_supervision(batch_inputs['occ_semantickitti_masked'],[1,1,1],np.array([len(batch_data_samples),256,256,32],dtype=np.int32))
            voxels_lvl0 = voxels_lvl0.reshape(len(batch_data_samples),-1)
        else:
            voxels_lvl0 = self.multiscale_supervision(batch_inputs['dense_occ_200'],[1,1,1],np.array([len(batch_data_samples),200,200,16],dtype=np.int32))
            voxels_lvl0 = voxels_lvl0.reshape(len(batch_data_samples),-1)
        
        final_vox_logits = []
        if self.occ3d:
            for i, data_sample in enumerate(batch_data_samples):
                data_sample.eval_ann_info['pts_semantic_mask'] = voxels_lvl0[i].cpu().numpy().astype(np.uint8)
                data_sample.eval_ann_info['occ3d_mask'] = lvl0_masks[i].cpu().numpy().astype(np.uint8)
                final_vox_logits.append(occ_ori_logits[i].reshape(-1,Cls))
        else:
            for i, data_sample in enumerate(batch_data_samples):
                data_sample.eval_ann_info['pts_semantic_mask'] = voxels_lvl0[i].cpu().numpy().astype(np.uint8)
                final_vox_logits.append(occ_ori_logits[i].reshape(-1,Cls))

        return self.postprocess_result(final_vox_logits, batch_data_samples)
    
    def postprocess_result(self, voxel_logits, batch_data_samples):
        
        for i in range(len(voxel_logits)):
            voxel_logit = voxel_logits[i]
            voxel_pred = voxel_logit.argmax(dim=1)
            batch_data_samples[i].set_data({
                'pts_seg_logits':
                PointData(**{'pts_seg_logits': voxel_logit}),
                'pred_pts_seg':
                PointData(**{'pts_semantic_mask': voxel_pred})
            })
        return batch_data_samples
    
    def aug_test(self, batch_inputs, batch_data_samples):
        pass

    def encode_decode(self, batch_inputs, batch_data_samples):
        pass
    