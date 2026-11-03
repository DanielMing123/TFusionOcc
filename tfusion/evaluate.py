# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from typing import Dict, Optional, Sequence

import mmcv
import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmdet3d.evaluation import seg_eval, SegMetric
from mmdet3d.registry import METRICS


@METRICS.register_module()
class EvalMetric(SegMetric):
    """3D semantic segmentation evaluation metric.

    Args:
        collect_device (str, optional): Device name used for collecting
            results from different ranks during distributed training.
            Must be 'cpu' or 'gpu'. Defaults to 'cpu'.
        prefix (str): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None.
        pklfile_prefix (str, optional): The prefix of pkl files, including
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Default: None.
        submission_prefix (str, optional): The prefix of submission data.
            If not specified, the submission data will not be generated.
            Default: None.
    """

    def __init__(self,
                 dist_eval = False,
                 sector_dist_eval = False,
                 use_occ3d = False,
                 use_semanticKITTI = False,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 pklfile_prefix: str = None,
                 submission_prefix: str = None,
                 **kwargs):
        self.dist_eval = dist_eval
        self.sector_dist_eval = sector_dist_eval
        self.use_occ3d = use_occ3d
        self.use_semanticKITTI = use_semanticKITTI
        if self.use_occ3d and self.use_semanticKITTI:
            raise ValueError(
                '`use_occ3d` and `use_semanticKITTI` cannot both be True.')
        if self.use_occ3d:
            self.occ_eval_metrics = Metric_mIoU_Occ3D(num_classes=18,
                                                      use_lidar_mask=False,
                                                      use_image_mask=True)
        self.pklfile_prefix = pklfile_prefix
        self.submission_prefix = submission_prefix
        super(SegMetric, self).__init__(
            prefix=prefix, collect_device=collect_device)

    def generate_mask(self,
                      grid,
                      min_dist,
                      max_dist,
                      voxel_size=0.5,
                      ego_x=99,
                      ego_y=99):
        min_dist_vox = min_dist / voxel_size
        max_dist_vox = max_dist / voxel_size

        # Create coordinate grid
        xs = np.arange(grid.shape[0])
        ys = np.arange(grid.shape[1])
        xx, yy = np.meshgrid(xs, ys, indexing='ij')

        # Compute distance in voxel units
        dist = np.sqrt((xx - ego_x)**2 + (yy - ego_y)**2)

        # Mask for 10–20m ring
        mask = (dist >= min_dist_vox) & (dist <= max_dist_vox)
        
        return mask
    
    def evaluation_semantic(self, 
                            gt_labels,
                            seg_preds,
                            label2cat,
                            ignore_index,
                            logger=None):
        assert len(seg_preds) == len(gt_labels)
        classes_num = len(label2cat)
        ret_dict = dict()
        results = []
        for i in range(len(gt_labels)):
            gt_i, pred_i = gt_labels[i].astype(np.int64), seg_preds[i].astype(np.int64)
            mask = (gt_i != ignore_index)
            score = np.zeros((classes_num, 3))
            for j in range(classes_num):
                if j == 0: #class 0 for geometry IoU
                    score[j][0] += ((gt_i[mask] != 0) * (pred_i[mask] != 0)).sum()
                    score[j][1] += (gt_i[mask] != 0).sum()
                    score[j][2] += (pred_i[mask] != 0).sum()
                else:
                    score[j][0] += ((gt_i[mask] == j) * (pred_i[mask] == j)).sum()
                    score[j][1] += (gt_i[mask] == j).sum()
                    score[j][2] += (pred_i[mask] == j).sum()
            
            results.append(score)
            
        results = np.stack(results, axis=0).mean(0)
        mean_ious = []
        for i in range(classes_num):
            tp = results[i, 0]
            p = results[i, 1]
            g = results[i, 2]
            union = p + g - tp
            mean_ious.append(tp / union)
        mean_ious = np.nan_to_num(mean_ious)
        for i in range(len(label2cat)):
            ret_dict[label2cat[i]] = mean_ious[i]
        ret_dict['mIoU'] = np.mean(np.array(mean_ious)[1:])
        
        return ret_dict
    
    def evaluation_semantic_fast(self,
                                gt_labels,
                                seg_preds,
                                label2cat,
                                ignore_index,
                                logger=None):
        assert len(seg_preds) == len(gt_labels)
        classes_num = len(label2cat)

        tp = np.zeros(classes_num, dtype=np.int64)
        gt_pos = np.zeros(classes_num, dtype=np.int64)
        pred_pos = np.zeros(classes_num, dtype=np.int64)

        for gt_i, pred_i in zip(gt_labels, seg_preds):
            gt_i = gt_i.astype(np.int64, copy=False)
            pred_i = pred_i.astype(np.int64, copy=False)

            mask = (gt_i != ignore_index)
            if not np.any(mask):
                continue

            g = gt_i[mask]
            p = pred_i[mask]

            # 先存 geometry(class0) 的统计（non-zero vs non-zero）
            g_nz = (g != 0)
            p_nz = (p != 0)
            tp0 = np.count_nonzero(g_nz & p_nz)
            gt0 = np.count_nonzero(g_nz)
            pr0 = np.count_nonzero(p_nz)

            # 类别统计（包含 label==0 的普通语义统计）
            gt_cnt = np.bincount(g, minlength=classes_num)
            pr_cnt = np.bincount(p, minlength=classes_num)

            m = (g == p)
            tp_cnt = np.bincount(g[m], minlength=classes_num) if np.any(m) else np.zeros(classes_num, dtype=np.int64)

            # 覆盖 class0 为 geometry 定义
            gt_cnt[0] = gt0
            pr_cnt[0] = pr0
            tp_cnt[0] = tp0

            gt_pos += gt_cnt
            pred_pos += pr_cnt
            tp += tp_cnt

        union = gt_pos + pred_pos - tp
        mean_ious = tp / union.astype(np.float64)
        mean_ious = np.nan_to_num(mean_ious, nan=0.0, posinf=0.0, neginf=0.0)

        ret_dict = {label2cat[i]: mean_ious[i] for i in range(classes_num)}
        ret_dict["mIoU"] = float(np.mean(mean_ious[1:]))  # 语义 mIoU 不含 class0
        return ret_dict
    
    def evaluation_semantic_chunked(self, gt_labels, seg_preds, label2cat, ignore_index, chunk_size=256, logger=None):
        assert len(seg_preds) == len(gt_labels)
        classes_num = len(label2cat)

        tp = np.zeros(classes_num, dtype=np.int64)
        gt_pos = np.zeros(classes_num, dtype=np.int64)
        pred_pos = np.zeros(classes_num, dtype=np.int64)

        n = len(gt_labels)
        for s in range(0, n, chunk_size):
            g_list, p_list = [], []
            for gt_i, pred_i in zip(gt_labels[s:s+chunk_size], seg_preds[s:s+chunk_size]):
                gt_i = gt_i.astype(np.int64, copy=False)
                pred_i = pred_i.astype(np.int64, copy=False)
                m = (gt_i != ignore_index)
                if np.any(m):
                    g_list.append(gt_i[m])
                    p_list.append(pred_i[m])

            if len(g_list) == 0:
                continue

            g = np.concatenate(g_list, axis=0)
            p = np.concatenate(p_list, axis=0)

            # geometry(class0)
            g_nz = (g != 0)
            p_nz = (p != 0)
            tp0 = np.count_nonzero(g_nz & p_nz)
            gt0 = np.count_nonzero(g_nz)
            pr0 = np.count_nonzero(p_nz)

            # 语义统计
            gt_cnt = np.bincount(g, minlength=classes_num).astype(np.int64, copy=False)
            pr_cnt = np.bincount(p, minlength=classes_num).astype(np.int64, copy=False)
            eq = (g == p)
            tp_cnt = np.bincount(g[eq], minlength=classes_num).astype(np.int64, copy=False) if np.any(eq) \
                    else np.zeros(classes_num, dtype=np.int64)

            # 覆盖 class0
            gt_cnt[0], pr_cnt[0], tp_cnt[0] = gt0, pr0, tp0

            gt_pos += gt_cnt
            pred_pos += pr_cnt
            tp += tp_cnt

        union = gt_pos + pred_pos - tp
        mean_ious = tp / union.astype(np.float64)
        mean_ious = np.nan_to_num(mean_ious, nan=0.0, posinf=0.0, neginf=0.0)

        ret_dict = {label2cat[i]: mean_ious[i] for i in range(classes_num)}
        ret_dict["mIoU"] = float(np.mean(mean_ious[1:]))
        return ret_dict
    
    def evaluation_semanticKITTI(self,
                                 gt_labels,
                                 seg_preds,
                                 label2cat,
                                 ignore_index):
        """Evaluate SemanticKITTI semantic scene completion predictions.

        Class 0 is the empty voxel class. Completion precision, recall and IoU
        therefore treat every class greater than 0 as occupied. Semantic IoUs
        and mIoU are computed over the 19 occupied classes. Ground-truth voxels
        marked with ``ignore_index`` are excluded, as required by the official
        SemanticKITTI completion protocol.
        """
        if len(seg_preds) != len(gt_labels):
            raise ValueError(
                'The number of SemanticKITTI predictions and labels must '
                f'match, but got {len(seg_preds)} and {len(gt_labels)}.')

        classes_num = len(label2cat)
        if classes_num < 2:
            raise ValueError(
                'SemanticKITTI evaluation requires an empty class and at '
                'least one occupied class.')

        confusion = np.zeros((classes_num, classes_num), dtype=np.int64)
        for sample_idx, (gt_i, pred_i) in enumerate(
                zip(gt_labels, seg_preds)):
            gt_i = np.asarray(gt_i, dtype=np.int64).reshape(-1)
            pred_i = np.asarray(pred_i, dtype=np.int64).reshape(-1)
            if gt_i.shape != pred_i.shape:
                raise ValueError(
                    'SemanticKITTI prediction and label shapes must match '
                    f'for sample {sample_idx}, but got {pred_i.shape} and '
                    f'{gt_i.shape}.')

            valid_mask = gt_i != ignore_index
            gt_i = gt_i[valid_mask]
            pred_i = pred_i[valid_mask]
            if gt_i.size == 0:
                continue

            invalid_gt = (gt_i < 0) | (gt_i >= classes_num)
            invalid_pred = (pred_i < 0) | (pred_i >= classes_num)
            if np.any(invalid_gt):
                invalid_values = np.unique(gt_i[invalid_gt])
                raise ValueError(
                    'SemanticKITTI ground truth contains labels outside '
                    f'[0, {classes_num - 1}]: {invalid_values.tolist()}')
            if np.any(invalid_pred):
                invalid_values = np.unique(pred_i[invalid_pred])
                raise ValueError(
                    'SemanticKITTI prediction contains labels outside '
                    f'[0, {classes_num - 1}]: {invalid_values.tolist()}')

            confusion += np.bincount(
                classes_num * gt_i + pred_i,
                minlength=classes_num**2).reshape(classes_num, classes_num)

        occupied_intersection = confusion[1:, 1:].sum(dtype=np.int64)
        predicted_occupied = confusion[:, 1:].sum(dtype=np.int64)
        target_occupied = confusion[1:, :].sum(dtype=np.int64)
        occupied_union = (predicted_occupied + target_occupied -
                          occupied_intersection)

        def safe_divide(numerator, denominator):
            if denominator == 0:
                return 0.0
            return float(numerator / denominator)

        precision = safe_divide(occupied_intersection, predicted_occupied)
        recall = safe_divide(occupied_intersection, target_occupied)
        completion_iou = safe_divide(occupied_intersection, occupied_union)

        semantic_intersection = np.diag(confusion)[1:]
        semantic_union = (confusion[1:, :].sum(axis=1) +
                          confusion[:, 1:].sum(axis=0) -
                          semantic_intersection)
        semantic_iou = np.divide(
            semantic_intersection,
            semantic_union,
            out=np.zeros(classes_num - 1, dtype=np.float64),
            where=semantic_union != 0)

        ret_dict = {
            'precision': precision,
            'recall': recall,
            'IoU': completion_iou,
        }
        for class_idx in range(1, classes_num):
            ret_dict[label2cat[class_idx]] = float(
                semantic_iou[class_idx - 1])
        ret_dict['mIoU'] = float(np.mean(semantic_iou))
        return ret_dict

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        if self.submission_prefix:
            self.format_results(results)
            return None

        if self.use_occ3d:
            for eval_ann, sinlge_pred_results in results:
                gt = eval_ann['pts_semantic_mask']
                gt[gt==18]=0  # others from 18->0
                cam_mask = eval_ann['occ3d_mask'].astype(bool)
                pred = sinlge_pred_results['pts_semantic_mask']
                pred[pred==0]=18 # empty 0->18
                pred[pred==17]=0 # others 17->0
                pred[pred==18]=17 # empty 18->17
                self.occ_eval_metrics.add_batch(pred,
                                                gt,
                                                None,
                                                cam_mask)
                
            ret_dict = self.occ_eval_metrics.count_miou()
        elif self.use_semanticKITTI:
            label2cat = self.dataset_meta['label2cat']
            ignore_index = self.dataset_meta['ignore_index']
            gt_semantic_masks = []
            pred_semantic_masks = []

            for eval_ann, single_pred_results in results:
                gt_semantic_masks.append(eval_ann['pts_semantic_mask'])
                pred_semantic_masks.append(
                    single_pred_results['pts_semantic_mask'])

            ret_dict = self.evaluation_semanticKITTI(
                gt_semantic_masks,
                pred_semantic_masks,
                label2cat,
                ignore_index)
        else:
            label2cat = self.dataset_meta['label2cat']
            ignore_index = self.dataset_meta['ignore_index']

            gt_semantic_masks = []
            pred_semantic_masks = []

            for eval_ann, sinlge_pred_results in results:
                gt_semantic_masks.append(eval_ann['pts_semantic_mask'])
                pred_semantic_masks.append(sinlge_pred_results['pts_semantic_mask'])

            if self.dist_eval:
                loop = 7
            elif self.sector_dist_eval:
                loop=5
            if self.dist_eval or self.sector_dist_eval:
                for i in range(loop):
                    gt_list, pred_list = [], []
                    for gt, pred in zip(gt_semantic_masks, pred_semantic_masks):
                        gt = gt.reshape(200,200,16)
                        pred = pred.reshape(200,200,16)
                        if i==0:
                            if self.sector_dist_eval:
                                mask = self.generate_mask(pred,0.0,10.0)
                                gt = gt[mask]
                                pred = pred[mask]
                            elif self.dist_eval:
                                gt = gt[59:139, 59:139, :]
                                pred = pred[59:139, 59:139, :]
                        elif i==1:
                            if self.sector_dist_eval:
                                mask = self.generate_mask(pred,10.0,20.0)
                                gt = gt[mask]
                                pred = pred[mask]
                            elif self.dist_eval:
                                gt = gt[49:149, 49:149, :]
                                pred = pred[49:149, 49:149, :]
                        elif i==2:
                            if self.sector_dist_eval:
                                mask = self.generate_mask(pred,20.0,30.0)
                                gt = gt[mask]
                                pred = pred[mask]
                            elif self.dist_eval:
                                gt = gt[39:159, 39:159, :]
                                pred = pred[39:159, 39:159, :]
                        elif i==3:
                            if self.sector_dist_eval:
                                mask = self.generate_mask(pred,30.0,40.0)
                                gt = gt[mask]
                                pred = pred[mask]
                            elif self.dist_eval:
                                gt = gt[29:169, 29:169, :]
                                pred = pred[29:169, 29:169, :]
                        elif i==4:
                            if self.sector_dist_eval:
                                mask = self.generate_mask(pred,40.0,50.0)
                                gt = gt[mask]
                                pred = pred[mask]
                            elif self.dist_eval:
                                gt = gt[19:179, 19:179, :]
                                pred = pred[19:179, 19:179, :]
                        elif i==5:
                            gt = gt[9:189, 9:189, :]
                            pred = pred[9:189, 9:189, :]
                        
                        gt_list.append(gt.flatten())
                        pred_list.append(pred.flatten())
                        
                    ret_dict = self.evaluation_semantic_chunked(gt_list, # list of gt, gt_semantic_masks
                                                                pred_list, # list of pred, pred_semantic_masks
                                                                label2cat,
                                                                ignore_index,
                                                                logger=logger)
                    print(ret_dict)
            else:
                ret_dict = self.evaluation_semantic_chunked(gt_semantic_masks,
                                                            pred_semantic_masks,
                                                            label2cat,
                                                            ignore_index,
                                                            logger=logger)
            
        return ret_dict
    
class Metric_mIoU_Occ3D():
    def __init__(self,
                 num_classes=18,
                 use_lidar_mask=False,
                 use_image_mask=False,
                 ):
        self.class_names = ['others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                            'driveable_surface', 'other_flat', 'sidewalk',
                            'terrain', 'manmade', 'vegetation', 'free']
        self.use_lidar_mask = use_lidar_mask
        self.use_image_mask = use_image_mask
        self.num_classes = num_classes

        self.point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        self.occupancy_size = [0.4, 0.4, 0.4]
        self.voxel_size = 0.4
        self.occ_xdim = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.occupancy_size[0])
        self.occ_ydim = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.occupancy_size[1])
        self.occ_zdim = int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.occupancy_size[2])
        self.voxel_num = self.occ_xdim * self.occ_ydim * self.occ_zdim
        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.cnt = 0

    def hist_info(self, n_cl, pred, gt):
        """
        build confusion matrix
        # empty classes:0
        non-empty class: 0-16
        free voxel class: 17

        Args:
            n_cl (int): num_classes_occupancy
            pred (1-d array): pred_occupancy_label
            gt (1-d array): gt_occupancu_label

        Returns:
            tuple:(hist, correctly number_predicted_labels, num_labelled_sample)
        """
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    def per_class_iu(self, hist):

        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def compute_mIoU(self, pred, label, n_classes):
        hist = np.zeros((n_classes, n_classes))
        new_hist, correct, labeled = self.hist_info(n_classes, pred.flatten(), label.flatten())
        hist += new_hist

        return hist

    def add_batch(self,semantics_pred,semantics_gt,mask_lidar,mask_camera):
        self.cnt += 1
        if self.use_image_mask:
            masked_semantics_gt = semantics_gt[mask_camera]
            masked_semantics_pred = semantics_pred[mask_camera]
        elif self.use_lidar_mask:
            masked_semantics_gt = semantics_gt[mask_lidar]
            masked_semantics_pred = semantics_pred[mask_lidar]
        else:
            masked_semantics_gt = semantics_gt
            masked_semantics_pred = semantics_pred

        _hist = self.compute_mIoU(masked_semantics_pred, masked_semantics_gt, self.num_classes)
        self.hist += _hist

    def count_miou(self):
        mIoU = self.per_class_iu(self.hist)
        res = dict()
        for idx, clas in enumerate(self.class_names):
            res[clas] = round(mIoU[idx] * 100, 2)   

        print(f'===> per class IoU of {self.cnt} samples:')
        for ind_class in range(self.num_classes-1):
            print(f'===> {self.class_names[ind_class]} - IoU = ' + str(round(mIoU[ind_class] * 100, 2)))

        print(f'===> mIoU of {self.cnt} samples: ' + str(round(np.nanmean(mIoU[:self.num_classes-1]) * 100, 2)))
        res['mIoU'] = round(np.nanmean(mIoU[:self.num_classes-1]) * 100, 2)
        self.cnt = 0
        self.hist = np.zeros((self.num_classes, self.num_classes))
        
        return res