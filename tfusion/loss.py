import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.runner.amp import autocast
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from .safe_ops import safe_sigmoid, safe_inverse_sigmoid

def CE_wo_softmax(pred, target, class_weights=None, ignore_index=255):
    pred = torch.clamp(pred, 1e-6, 1. - 1e-6)
    loss = F.nll_loss(torch.log(pred), target, class_weights, ignore_index=ignore_index)
    return loss

def BCE_Occupy(pred, target):
    B = pred.shape[0]
    gt = torch.zeros_like(target)
    gt[target!=0] = 1.0
    gt = gt.reshape(B, -1).float()
    loss = F.binary_cross_entropy(pred, gt)
    
    return loss

def cam_init_loss(cam_pts, cam_logits, gt, pc_range, voxel_size):
    finite_logits = torch.isfinite(cam_logits)
    if not finite_logits.all().item():
        nonfinite_count = (~finite_logits).sum().item()
        raise FloatingPointError(
            'cam_logits contains '
            f'{nonfinite_count}/{cam_logits.numel()} non-finite values '
            'before camera initialization BCE')

    pc_range = torch.tensor(pc_range).cuda() 
    voxel_size = torch.tensor(voxel_size).cuda()
    
    oob_mask = (cam_pts[..., 0] < pc_range[0]) | (cam_pts[..., 0] >= pc_range[3]) | \
               (cam_pts[..., 1] < pc_range[1]) | (cam_pts[..., 1] >= pc_range[4]) | \
               (cam_pts[..., 2] < pc_range[2]) | (cam_pts[..., 2] >= pc_range[5])
    anchor_idx = (cam_pts - pc_range[:3].view(1, 1, 1, 1, 1, 3)) / voxel_size
    anchor_idx = anchor_idx.to(torch.int)
    anchor_idx[..., 0].clamp_(0, gt.shape[1] - 1)
    anchor_idx[..., 1].clamp_(0, gt.shape[2] - 1)
    anchor_idx[..., 2].clamp_(0, gt.shape[3] - 1)

    anchor_occ = torch.stack([occ[idx[..., 0], idx[..., 1], idx[..., 2]] for occ, idx in zip(gt, anchor_idx)])
    anchor_occ[oob_mask] = 0
    
    anchor_gt = (anchor_occ != 0) 
    anchor_gt = torch.cat([anchor_gt, ~torch.any(anchor_gt, dim=-1, keepdim=True)], dim=-1)
    
    cam_act = torch.softmax(cam_logits, dim=-1)
    loss = nn.functional.binary_cross_entropy(cam_act, anchor_gt.float())
    
    return loss
    

@autocast('cuda',torch.float32)
def geo_scal_loss(pred, ssc_target, semantic=True):

    # Get softmax probabilities
    if semantic:
        pred = F.softmax(pred, dim=1)

        # Compute empty and nonempty probabilities
        empty_probs = pred[:, 0, :, :, :]
    else:
        empty_probs = 1 - torch.sigmoid(pred)
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    mask = ssc_target != 255 # 255 noise
    nonempty_target = ssc_target != 0 # 0 empty
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    recall = intersection / nonempty_target.sum()
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
    return (
        (-torch.ones_like(precision)*torch.log(precision) - (1-torch.ones_like(precision))*torch.log(1-precision)) + \
        (-torch.ones_like(recall)*torch.log(recall) - (1-torch.ones_like(recall))*torch.log(1-recall)) + \
        (-torch.ones_like(spec)*torch.log(spec) - (1-torch.ones_like(spec))*torch.log(1-spec))
    )

@autocast('cuda',torch.float32)
def sem_scal_loss(pred, ssc_target):
    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)
    loss = 0
    count = 0
    mask = ssc_target != 255
    n_classes = pred.shape[1]
    for i in range(0, n_classes):

        # Get probability of class i
        p = pred[:, i, :, :, :]

        # Remove unknown voxels
        target_ori = ssc_target
        p = p[mask]
        target = ssc_target[mask]

        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0
        completion_target_ori = torch.ones_like(target_ori).float()
        completion_target_ori[target_ori != i] = 0
        if torch.sum(completion_target) > 0:
            count += 1.0
            nominator = torch.sum(p * completion_target)
            loss_class = 0
            if torch.sum(p) > 0:
                precision = nominator / (torch.sum(p))
                loss_precision = -torch.ones_like(precision)*torch.log(precision) - (1-torch.ones_like(precision))*torch.log(1-precision)
                loss_class += loss_precision
            if torch.sum(completion_target) > 0:
                recall = nominator / (torch.sum(completion_target))
                loss_recall = -torch.ones_like(recall)*torch.log(recall) - (1-torch.ones_like(recall))*torch.log(1-recall)
                loss_class += loss_recall
            if torch.sum(1 - completion_target) > 0:
                specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                    torch.sum(1 - completion_target)
                )
                loss_specificity = -torch.ones_like(specificity)*torch.log(specificity) - (1-torch.ones_like(specificity))*torch.log(1-specificity)
                loss_class += loss_specificity
            loss += loss_class
    return loss / count

@MODELS.register_module()
class DepthLoss(BaseModule):
    def __init__(self,
                 depth_channels,
                 downsample_factor,
                 dbound):
        super(DepthLoss, self).__init__()
        self.depth_channels = depth_channels
        self.downsample_factor = downsample_factor
        self.dbound = dbound
          
          
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

        return gt_depths.float()
    

    def forward(self, depth_preds, depth_labels):
        loss_dict = dict()
        level = 0
        for depth_pred, downsample_factor in zip(depth_preds, self.downsample_factor):
            B,N,C,H,W = depth_pred.shape
            depth_pred = depth_pred.view(B*N,C,H,W).float()
            flat_depth_labels = self.get_downsampled_gt_depth(depth_labels, downsample_factor)
            depth_pred = depth_pred.permute(0, 2, 3, 1).contiguous().view(-1, self.depth_channels)
            fg_mask = torch.max(flat_depth_labels, dim=1).values > 0.0

            depth_loss = F.binary_cross_entropy(depth_pred[fg_mask],
                                                flat_depth_labels[fg_mask],
                                                reduction='none').sum() / max(1.0, fg_mask.sum())
            loss_dict[f'depth_{level}_loss'] = 0.5 * depth_loss
            level += 1
            
        return loss_dict 

