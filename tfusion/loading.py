# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional, Union
import copy
import mmcv
import numpy as np
import torch
import os
import mmengine
from mmcv.transforms.base import BaseTransform
from mmengine.fileio import get
from mmdet3d.structures.points import BasePoints, get_points_type
from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles
from mmdet3d.registry import TRANSFORMS
from nuscenes.utils.data_classes import RadarPointCloud
import pykitti.utils as utils
from nuscenes.utils.geometry_utils import view_points
import cv2

@TRANSFORMS.register_module()
class SemanticKITTI_Image_Load(LoadMultiViewImageFromFiles):
    def transform(self, result: dict) -> Optional[dict]:
        calib_filepath = result['calib_path']
        filedata = utils.read_calib_file(calib_filepath)
        P_rect_20 = np.asarray(filedata['P2'], dtype=np.float64).reshape(3, 4)
        T_cam0_velo = np.eye(4, dtype=np.float64)
        T_cam0_velo[:3, :] = np.asarray(
            filedata['Tr'], dtype=np.float64).reshape(3, 4)

        K_P2 = np.eye(4, dtype=np.float64)
        K_P2[:3, :3] = P_rect_20[:3, :3]
        T2 = np.eye(4, dtype=np.float64)
        T2[:3, 3] = np.linalg.solve(P_rect_20[:3, :3],
                                    P_rect_20[:3, 3])

        T_cam2_velo = T2 @ T_cam0_velo
        lidar2img = K_P2 @ T_cam2_velo
        result['lidar2cam'] = np.stack([T_cam2_velo], axis=0)
        result['cam2img'] = np.stack([K_P2], axis=0)
        result['lidar2img'] = np.stack([lidar2img], axis=0)
        img_byte = get(result['img_path'], backend_args=self.backend_args) 
        img = mmcv.imfrombytes(img_byte, flag=self.color_type)
        result['img'] = [img]
        result['img_shape'] = (img.shape[0], img.shape[1])
        return result

@TRANSFORMS.register_module()
class LoadSemanticKITTI_Occupancy(BaseTransform):
    def get_remap_lut(self, label_map):
        maxkey = max(label_map.keys())

        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(label_map.keys())] = list(label_map.values())

        # in completion we have to distinguish empty and invalid voxels.
        # Important: For voxels 0 corresponds to "empty" and not "unlabeled".
        remap_lut[remap_lut == 0] = 255  # map 0 to 'invalid'
        remap_lut[0] = 0  # only 'empty' stays 'empty'.

        return remap_lut
        
    
    def unpack(self, compressed):
        ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
        uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
        uncompressed[::8] = compressed[:] >> 7 & 1
        uncompressed[1::8] = compressed[:] >> 6 & 1
        uncompressed[2::8] = compressed[:] >> 5 & 1
        uncompressed[3::8] = compressed[:] >> 4 & 1
        uncompressed[4::8] = compressed[:] >> 3 & 1
        uncompressed[5::8] = compressed[:] >> 2 & 1
        uncompressed[6::8] = compressed[:] >> 1 & 1
        uncompressed[7::8] = compressed[:] & 1

        return uncompressed
    
    def transform(self, result: dict) -> dict:
        gt = np.fromfile(result['voxel_gt_path'],dtype=np.uint16).astype(np.float32)
        invalid = np.fromfile(result['voxel_invalid_path'], dtype=np.uint8)
        invalid = self.unpack(invalid)
        remap_lut = self.get_remap_lut(result['label_mapping'])
        gt = remap_lut[gt.astype(np.uint16)].astype(np.float32)  # Remap 20 classes semanticKITTI SSC
        
        gt[np.isclose(invalid, 1)] = 255  # Setting to unknown all voxels marked on invalid mask...
        gt_masked = gt.reshape([256, 256, 32])
        gt_masked = torch.from_numpy(gt_masked)
        idx_masked = torch.where(gt_masked > 0)
        label_masked = gt_masked[idx_masked[0],idx_masked[1],idx_masked[2]]
        semantickitti_occ_masked = torch.stack([idx_masked[0],idx_masked[1],idx_masked[2],label_masked],dim=1).long()
        
        # result['occ_semantickitti'] = semantickitti_occ
        result['occ_semantickitti_masked'] = semantickitti_occ_masked
        return result
        
    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str

@TRANSFORMS.register_module()
class LoadSemanticKITTI_Lidar(BaseTransform):
    def __init__(self,
                pc_range=None):
       self.pc_range = pc_range
    
    def transform(self, result: dict) -> dict:
        lidar_path = result['voxel_gt_path'].split('voxels')[0] + 'velodyne' + result['voxel_gt_path'].split('voxels')[1].split('.label')[0] + '.bin'
        pts = utils.load_velo_scan(lidar_path)
        pts = torch.from_numpy(pts)
        idx = torch.where((pts[:,0] > self.pc_range[0])
                        & (pts[:,1] > self.pc_range[1])
                        & (pts[:,2] > self.pc_range[2])
                        & (pts[:,0] < self.pc_range[3])
                        & (pts[:,1] < self.pc_range[4])
                        & (pts[:,2] < self.pc_range[5]))
        pts = pts[idx]
        result['points'] = torch.cat([pts, torch.zeros((pts.size(0),1))], dim=-1)
        
        return result
        
    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str

@TRANSFORMS.register_module()
class BEVLoadMultiViewImageFromFiles(LoadMultiViewImageFromFiles):
    """Load multi channel images from a list of separate channel files.

    ``BEVLoadMultiViewImageFromFiles`` adds the following keys for the
    convenience of view transforms in the forward:
        - 'cam2lidar'
        - 'lidar2img'

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        num_views (int): Number of view in a frame. Defaults to 5.
        num_ref_frames (int): Number of frame in loading. Defaults to -1.
        test_mode (bool): Whether is test mode in loading. Defaults to False.
        set_default_scale (bool): Whether to set default scale.
            Defaults to True.
    """

    def transform(self, results: dict) -> Optional[dict]:
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
            Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename, cam2img, lidar2cam, lidar2img, ego2img = [], [], [], [], []
        for _, cam_item in results['images'].items():
            filename.append(cam_item['img_path'])
            lidar2cam.append(cam_item['lidar2cam'])
            
            ego2cam_array = np.linalg.inv(np.array(cam_item['cam2ego'],dtype=np.float64))
            lidar2cam_array = np.array(cam_item['lidar2cam'],dtype=np.float64)
            cam2img_array = np.eye(4).astype(np.float64)
            cam2img_array[:3, :3] = np.array(cam_item['cam2img'],dtype=np.float64)
            cam2img.append(cam2img_array)
            lidar2img.append(cam2img_array @ lidar2cam_array)
            ego2img.append(cam2img_array @ ego2cam_array)

        results['img_path'] = filename
        results['cam2img'] = np.stack(cam2img, axis=0)
        results['lidar2cam'] = np.stack(lidar2cam, axis=0)
        results['lidar2img'] = np.stack(lidar2img, axis=0)
        results['ego2img'] = np.stack(ego2img, axis=0)
        results['ori_cam2img'] = copy.deepcopy(results['cam2img'])

        # img is of shape (h, w, c, num_views)
        # h and w can be different for different views
        img_bytes = [
            get(name, backend_args=self.backend_args) for name in filename
        ]
        # gbr follow tpvformer
        imgs = [
            mmcv.imfrombytes(img_byte, flag=self.color_type)
            for img_byte in img_bytes
        ]
        # handle the image with different shape
        img_shapes = np.stack([img.shape for img in imgs], axis=0)
        img_shape_max = np.max(img_shapes, axis=0)
        img_shape_min = np.min(img_shapes, axis=0)
        assert img_shape_min[-1] == img_shape_max[-1]
        if not np.all(img_shape_max == img_shape_min):
            pad_shape = img_shape_max[:2]
        else:
            pad_shape = None
        if pad_shape is not None:
            imgs = [
                mmcv.impad(img, shape=pad_shape, pad_val=0) for img in imgs
            ]
        img = np.stack(imgs, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape[:2]
        if self.set_default_scale:
            results['scale_factor'] = 0.2 # 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['num_views'] = self.num_views
        results['num_ref_frames'] = self.num_ref_frames
        return results

def get_corruption_path_cam(corruption_root, corruption, severity, filepath):
    folder, filename = os.path.split(filepath)
    _, subfolder = os.path.split(folder)
    return os.path.join(corruption_root, corruption, severity, subfolder, filename)

@TRANSFORMS.register_module()
class Custom_LoadMultiViewImageFromFiles(BaseTransform):
    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'unchanged',
                 backend_args: Optional[dict] = None,
                 num_views: int = 6,
                 corruption=None, 
                 severity=None, 
                 corruption_root=None,
                 num_ref_frames: int = -1,
                 test_mode: bool = False,
                 set_default_scale: bool = True) -> None:
        self.to_float32 = to_float32
        self.color_type = color_type
        self.backend_args = backend_args
        self.num_views = num_views
        # num_ref_frames is used for multi-sweep loading
        self.num_ref_frames = num_ref_frames
        # when test_mode=False, we randomly select previous frames
        # otherwise, select the earliest one
        self.test_mode = test_mode
        self.set_default_scale = set_default_scale
        self.corruption = corruption
        self.severity = severity
        self.corruption_root = corruption_root
        if corruption is not None:
            assert severity in ['easy', 'mid', 'hard'], f"Specify a severity of corruption benchmark, now {severity}"
            assert corruption_root is not None, f"When benchmark corruption, specify nuScenes-C-Cam root"
        
    def transform(self, results: dict) -> Optional[dict]:
        # TODO: consider split the multi-sweep part out of this pipeline
        # Derive the mask and transform for loading of multi-sweep data
        if self.num_ref_frames > 0:
            # init choice with the current frame
            init_choice = np.array([0], dtype=np.int64)
            num_frames = len(results['img_filename']) // self.num_views - 1
            if num_frames == 0:  # no previous frame, then copy cur frames
                choices = np.random.choice(
                    1, self.num_ref_frames, replace=True)
            elif num_frames >= self.num_ref_frames:
                # NOTE: suppose the info is saved following the order
                # from latest to earlier frames
                if self.test_mode:
                    choices = np.arange(num_frames - self.num_ref_frames,
                                        num_frames) + 1
                # NOTE: +1 is for selecting previous frames
                else:
                    choices = np.random.choice(
                        num_frames, self.num_ref_frames, replace=False) + 1
            elif num_frames > 0 and num_frames < self.num_ref_frames:
                if self.test_mode:
                    base_choices = np.arange(num_frames) + 1
                    random_choices = np.random.choice(
                        num_frames,
                        self.num_ref_frames - num_frames,
                        replace=True) + 1
                    choices = np.concatenate([base_choices, random_choices])
                else:
                    choices = np.random.choice(
                        num_frames, self.num_ref_frames, replace=True) + 1
            else:
                raise NotImplementedError
            choices = np.concatenate([init_choice, choices])
            select_filename = []
            for choice in choices:
                select_filename += results['img_filename'][choice *
                                                           self.num_views:
                                                           (choice + 1) *
                                                           self.num_views]
            results['img_filename'] = select_filename
            for key in ['cam2img', 'lidar2cam']:
                if key in results:
                    select_results = []
                    for choice in choices:
                        select_results += results[key][choice *
                                                       self.num_views:(choice +
                                                                       1) *
                                                       self.num_views]
                    results[key] = select_results
            for key in ['ego2global']:
                if key in results:
                    select_results = []
                    for choice in choices:
                        select_results += [results[key][choice]]
                    results[key] = select_results
            # Transform lidar2cam to
            # [cur_lidar]2[prev_img] and [cur_lidar]2[prev_cam]
            for key in ['lidar2cam']:
                if key in results:
                    # only change matrices of previous frames
                    for choice_idx in range(1, len(choices)):
                        pad_prev_ego2global = np.eye(4)
                        prev_ego2global = results['ego2global'][choice_idx]
                        pad_prev_ego2global[:prev_ego2global.
                                            shape[0], :prev_ego2global.
                                            shape[1]] = prev_ego2global
                        pad_cur_ego2global = np.eye(4)
                        cur_ego2global = results['ego2global'][0]
                        pad_cur_ego2global[:cur_ego2global.
                                           shape[0], :cur_ego2global.
                                           shape[1]] = cur_ego2global
                        cur2prev = np.linalg.inv(pad_prev_ego2global).dot(
                            pad_cur_ego2global)
                        for result_idx in range(choice_idx * self.num_views,
                                                (choice_idx + 1) *
                                                self.num_views):
                            results[key][result_idx] = \
                                results[key][result_idx].dot(cur2prev)
        # Support multi-view images with different shapes
        # TODO: record the origin shape and padded shape
        filename, cam2img, lidar2cam = [], [], []
        for _, cam_item in results['images'].items():
            filename.append(cam_item['img_path'])
            cam2img.append(cam_item['cam2img'])
            lidar2cam.append(cam_item['lidar2cam'])
        
        if self.corruption != 'Clean':
            corrupt_filename = []
            for orig_filename in filename:
                name = os.path.split(orig_filename)[1]
                subfolder = os.path.split(os.path.split(orig_filename)[0])[1]
                subfolder_name = os.path.join(subfolder, name)
                file_path = get_corruption_path_cam(self.corruption_root, self.corruption, self.severity, subfolder_name)
                corrupt_filename.append(file_path)
            
            results['filename'] = corrupt_filename
        else:
            results['filename'] = filename
        results['cam2img'] = cam2img
        results['lidar2cam'] = lidar2cam

        results['ori_cam2img'] = copy.deepcopy(results['cam2img'])

        # img is of shape (h, w, c, num_views)
        # h and w can be different for different views
        img_bytes = [
            get(name, backend_args=self.backend_args) for name in results['filename'] # filename
        ]
        imgs = [
            mmcv.imfrombytes(img_byte, flag=self.color_type)
            for img_byte in img_bytes
        ]
        # handle the image with different shape
        img_shapes = np.stack([img.shape for img in imgs], axis=0)
        img_shape_max = np.max(img_shapes, axis=0)
        img_shape_min = np.min(img_shapes, axis=0)
        assert img_shape_min[-1] == img_shape_max[-1]
        if not np.all(img_shape_max == img_shape_min):
            pad_shape = img_shape_max[:2]
        else:
            pad_shape = None
        if pad_shape is not None:
            imgs = [
                mmcv.impad(img, shape=pad_shape, pad_val=0) for img in imgs
            ]
        img = np.stack(imgs, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape[:2]
        if self.set_default_scale:
            results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['num_views'] = self.num_views
        results['num_ref_frames'] = self.num_ref_frames
        return results

def get_corruption_path_lidar(corruption_root, corruption, severity, filepath):
    folder, filename = os.path.split(filepath)
    _, subfolder = os.path.split(folder)
    return os.path.join(corruption_root, corruption, severity, 'samples', subfolder, filename)

@TRANSFORMS.register_module()
class Custom_LoadPointsFromFile(BaseTransform):
    def __init__(self,
                 coord_type: str,
                 load_dim: int = 6,
                 use_dim: Union[int, List[int]] = [0, 1, 2],
                 shift_height: bool = False,
                 use_color: bool = False,
                 norm_intensity: bool = False,
                 norm_elongation: bool = False,
                 corruption=None, 
                 severity=None, 
                 corruption_root=None,
                 backend_args: Optional[dict] = None) -> None:
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.norm_intensity = norm_intensity
        self.norm_elongation = norm_elongation
        self.backend_args = backend_args
        self.corruption = corruption
        self.severity = severity
        self.corruption_root = corruption_root
        if corruption is not None:
            assert severity in ['light', 'moderate', 'heavy'], f"Specify a severity of corruption benchmark, now {severity}"
            assert corruption_root is not None, f"When benchmark corruption, specify nuScenes-C-Lidar root"

    def _load_points(self, pts_filename: str) -> np.ndarray:
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        try:
            pts_bytes = get(pts_filename, backend_args=self.backend_args)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmengine.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def transform(self, results: dict) -> dict:
        """Method to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
            Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_file_path = results['lidar_points']['lidar_path']
        if self.corruption != 'Clean':
            pts_file_path = get_corruption_path_lidar(self.corruption_root, self.corruption, self.severity, pts_file_path)
        points = self._load_points(pts_file_path)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        if self.norm_intensity:
            assert len(self.use_dim) >= 4, \
                f'When using intensity norm, expect used dimensions >= 4, got {len(self.use_dim)}'  # noqa: E501
            points[:, 3] = np.tanh(points[:, 3])
        if self.norm_elongation:
            assert len(self.use_dim) >= 5, \
                f'When using elongation norm, expect used dimensions >= 5, got {len(self.use_dim)}'  # noqa: E501
            points[:, 4] = np.tanh(points[:, 4])
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'backend_args={self.backend_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        repr_str += f'norm_intensity={self.norm_intensity})'
        repr_str += f'norm_elongation={self.norm_elongation})'
        return repr_str

@TRANSFORMS.register_module()
class SegLabelMapping(BaseTransform):
    """Map original semantic class to valid category ids.

    Required Keys:

    - seg_label_mapping (np.ndarray)
    - pts_semantic_mask (np.ndarray)

    Added Keys:

    - points (np.float32)

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).
    """

    def transform(self, results: dict) -> dict:
        """Call function to map original semantic class to valid category ids.

        Args:
            results (dict): Result dict containing point semantic masks.

        Returns:
            dict: The result dict containing the mapped category ids.
            Updated key and value are described below.

                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
        """
        assert 'pts_semantic_mask' in results
        pts_semantic_mask = results['pts_semantic_mask']

        assert 'seg_label_mapping' in results
        label_mapping = results['seg_label_mapping']
        converted_pts_sem_mask = np.vectorize(
            label_mapping.__getitem__, otypes=[np.uint8])(
                pts_semantic_mask)

        results['pts_semantic_mask'] = converted_pts_sem_mask

        # 'eval_ann_info' will be passed to evaluator
        if 'eval_ann_info' in results:
            assert 'pts_semantic_mask' in results['eval_ann_info']
            results['eval_ann_info']['pts_semantic_mask'] = \
                converted_pts_sem_mask

        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str

@TRANSFORMS.register_module()
class LoadOccupancy(BaseTransform):
    def __init__(self,
                use_occ3d=False):
       self.use_occ3d = use_occ3d
    
    def transform(self, results: dict) -> dict:
        ####### load ego2img matrices ##########
        lidar2img, ego2img = [],[]
        for _, cam_item in results['images'].items():
            ego2cam_array = np.linalg.inv(np.array(cam_item['cam2ego'],dtype=np.float64))
            lidar2cam_array = np.array(cam_item['lidar2cam'],dtype=np.float64)
            cam2img_array = np.eye(4).astype(np.float64)
            cam2img_array[:3, :3] = np.array(cam_item['cam2img'],dtype=np.float64)
            lidar2img.append(cam2img_array @ lidar2cam_array)
            ego2img.append(cam2img_array @ ego2cam_array)
        results['ego2img'] = np.stack(ego2img, axis=0)
        results['lidar2img'] = np.stack(lidar2img, axis=0)
        
        if self.use_occ3d:
            ####### load Occ3D Dataset #########   
            occ_3d_folder = './data/nuscenes/Occ3D'
            occ_3d_path = os.path.join(occ_3d_folder, results['token'], 'labels.npz')
            occ_3d = np.load(occ_3d_path)
            occ_3d_semantic = occ_3d['semantics']
            occ_3d_cam_mask = occ_3d['mask_camera']
            #### Process occ3d_semantic ###### 17-> empty, 18 -> others
            occ_3d_semantic[occ_3d_semantic==0]=18 # others from 0->18  
            x, y, z = np.where(occ_3d_semantic > 0)
            label = occ_3d_semantic[x,y,z]
            occ_3d_labels = torch.from_numpy(np.stack([x,y,z,label],axis=1))
            #### Process occ3d mask #######
            x_mask, y_mask, z_mask = np.where(occ_3d_cam_mask > 0)
            masks = occ_3d_cam_mask[x_mask,y_mask,z_mask]
            occ_3d_mask = torch.from_numpy(np.stack([x_mask,y_mask,z_mask,masks],axis=1))
            results['occ_3d_labels'] = occ_3d_labels
            results['occ_3d_mask'] = occ_3d_mask
        else:
            occ_file_name = results['lidar_points']['lidar_path'].split('/')[-1] + '.npy'
            occ_200_folder = './data/nuscenes/occ_samples' # /workdir/MulVision
            occ_200_path = os.path.join(occ_200_folder, occ_file_name)
            occ_200 = np.load(occ_200_path)
            occ_200[:,3][occ_200[:,3]==0]=255
            occ_200 = torch.from_numpy(occ_200)
            results['occ_200'] = occ_200
            
        return results
        
    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str

@TRANSFORMS.register_module()
class LoadDepthGT(BaseTransform):
    def transform(self, results: dict) -> dict:
        l2cs = np.array(results['lidar2cam']) # lidar 2 camera
        c2is = np.array(results['cam2img']) # camera 2 img
        pts_l = results['points'][:,:3].numpy()
        ones = np.ones((pts_l.shape[0],1))
        pts_l = np.concatenate([pts_l,ones],axis=1)
        depth_gt = []
        for l2c, c2i in zip(l2cs,c2is):
            pts_c = np.matmul(l2c,pts_l.T)
            depth = pts_c[2,:]
            coloring = depth
            pts_i = view_points(pts_c[:3,:], c2i, normalize=True)
            mask = np.ones(depth.shape[0], dtype=bool)
            mask = np.logical_and(mask, depth > 0.0)
            mask = np.logical_and(mask, pts_i[0, :] > 1)
            mask = np.logical_and(mask, pts_i[0, :] < results['img_shape'][1] - 1)
            mask = np.logical_and(mask, pts_i[1, :] > 1)
            mask = np.logical_and(mask, pts_i[1, :] < results['img_shape'][0] - 1)
            pts_i = pts_i[:, mask]
            coloring = coloring[mask]
            pts_i = np.concatenate([pts_i[:2, :].T, coloring[:, None]], axis=1).astype(np.float32)
            
            depth_coords = pts_i[:, :2].astype(np.int16)
            depth_map = np.zeros(results['img_shape']) # (900,1600)
            valid_mask = ((depth_coords[:, 1] < results['img_shape'][0])
                        & (depth_coords[:, 0] < results['img_shape'][1])
                        & (depth_coords[:, 1] >= 0)
                        & (depth_coords[:, 0] >= 0))
            depth_map[depth_coords[valid_mask, 1],
                      depth_coords[valid_mask, 0]] = pts_i[valid_mask, 2]
            depth_map = torch.Tensor(depth_map)
            depth_gt.append(depth_map)
            
        depth_gt = torch.stack(depth_gt)
        depth_gt = depth_gt.unsqueeze(1)
        results['depth_gt'] = depth_gt
        
        return results
        
    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str
    

@TRANSFORMS.register_module()
class Cart2Polar(BaseTransform):
    def __init__(self, 
                 grid_size_pol, 
                 grid_size_occ,
                 voxel_size,
                 pc_range,
                 pol_range,
                 use_occ3d=False):
        super(Cart2Polar, self).__init__()
        self.grid_size_pol = torch.tensor(grid_size_pol)
        self.grid_size_occ = np.array(grid_size_occ)
        self.voxel_size = np.array(voxel_size)
        self.pc_range = np.array(pc_range)
        self.min_bound_pol = torch.tensor(pol_range[:3])
        self.max_bound_pol = torch.tensor(pol_range[3:])
        self.intervals_pol = (self.max_bound_pol - self.min_bound_pol) / self.grid_size_pol
        self.use_occ3d = use_occ3d
        
        
    def cart2polar(self, input_xyz):
        rho = torch.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
        phi = torch.arctan2(input_xyz[:, 1], input_xyz[:, 0])
        return torch.stack((rho, phi, input_xyz[:, 2]), axis=1)
    
    def transform(self, results: dict) -> dict:
        if type(results['points']) == torch.Tensor:
            xyz, feat = results['points'][:,:3], results['points'][:,3:]
        else:
            xyz, feat = results['points'].tensor[:,:3], results['points'].tensor[:,3:]
        ########### Spatial Outlier ##################
        # outlier_ratio = 0.02  # severe/severity-5 benchmark setting
        # outlier_std = 3.0     # meters

        # num_outliers = round(xyz.shape[0] * outlier_ratio)

        # if num_outliers > 0:
        #     outlier_idx = torch.randperm(
        #         xyz.shape[0], device=xyz.device
        #     )[:num_outliers]

        #     xyz = xyz.clone()
        #     xyz[outlier_idx] += torch.randn(
        #         (num_outliers, 3),
        #         device=xyz.device,
        #         dtype=xyz.dtype,
        #     ) * outlier_std
        ##############################################
        ########### Calibration error ################
        # Medium calibration error.
        # Approximate severity-3 settings: 0.012 rad rotation, 0.12 m translation.
        # rot_std = 0.02  # mid 0.012, sever 0.02
        # trans_std = 0.2 # mid 0.12,  sever 0.2

        # roll, pitch, yaw = np.random.normal(0.0, rot_std, size=3)
        # tx, ty, tz = np.random.normal(0.0, trans_std, size=3)

        # cx, sx = np.cos(roll), np.sin(roll)
        # cy, sy = np.cos(pitch), np.sin(pitch)
        # cz, sz = np.cos(yaw), np.sin(yaw)

        # Rx = np.array([
        #     [1, 0, 0],
        #     [0, cx, -sx],
        #     [0, sx, cx],
        # ], dtype=np.float64)

        # Ry = np.array([
        #     [cy, 0, sy],
        #     [0, 1, 0],
        #     [-sy, 0, cy],
        # ], dtype=np.float64)

        # Rz = np.array([
        #     [cz, -sz, 0],
        #     [sz, cz, 0],
        #     [0, 0, 1],
        # ], dtype=np.float64)

        # calib_error = np.eye(4, dtype=np.float64)
        # calib_error[:3, :3] = Rz @ Ry @ Rx
        # calib_error[:3, 3] = [tx, ty, tz]
        # results['lidar2img'] = (np.asarray(results['lidar2img'], dtype=np.float64) @ calib_error)
        ####################################################################
        ########### Randomly drop approximately 20% / 40% of points ##############
        # keep_mask = torch.rand(xyz.shape[0], device=xyz.device) >= 0.4
        # xyz = xyz[keep_mask]
        # feat = feat[keep_mask]
        ####################################################################
        if self.use_occ3d:
            lidar2ego = torch.tensor(results['lidar_points']['lidar2ego'])
            ones = torch.ones((xyz.shape[0],1))
            xyz_homo = torch.cat([xyz,ones],axis=1)
            xyz = torch.matmul(lidar2ego,xyz_homo.T).T[:, :3]
            xyz_mask = (xyz[..., 0] < self.pc_range[0]) | (xyz[..., 0] >= self.pc_range[3]) | \
                        (xyz[..., 1] < self.pc_range[1]) | (xyz[..., 1] >= self.pc_range[4]) | \
                        (xyz[..., 2] < self.pc_range[2]) | (xyz[..., 2] >= self.pc_range[5])
            xyz = xyz[~xyz_mask]
            feat = feat[~xyz_mask]
        xyz_pol = self.cart2polar(xyz)
        xyz_pol_grid = torch.clip(xyz_pol, self.min_bound_pol, self.max_bound_pol - 1e-3)
        grid_ind = (torch.floor((xyz_pol_grid - self.min_bound_pol) / self.intervals_pol)).to(torch.int32)
        voxel_centers = (grid_ind.to(torch.float32) + 0.5) * self.intervals_pol + self.min_bound_pol
        return_xyz = xyz_pol - voxel_centers
        return_feat = torch.cat((return_xyz, xyz_pol, xyz, feat), axis=1)
        
        all_locs = []
        for i in range(4): 
            grid_size_occ = self.grid_size_occ // (2**i)
            voxel_size = self.voxel_size * (2**i)
            dim_array = np.ones(len(grid_size_occ) + 1, np.int32)
            dim_array[0] = -1
            sample_loc = ((np.indices(grid_size_occ) + 0.5) * voxel_size.reshape(dim_array) + self.pc_range[:3].reshape(dim_array)).reshape(3, -1).transpose(1,0)
            sample_loc = torch.tensor(sample_loc)
            all_locs.append(sample_loc)
            
        results['pts_pol_vox_coords'] = grid_ind
        results['pts_pol_feat'] = return_feat
        results['sample_loc'] = all_locs
        
        return results
    
    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str

@TRANSFORMS.register_module()
class LoadRadarPointsMultiSweeps(BaseTransform):
   """Load radar points from multiple sweeps.
   This is usually used for nuScenes dataset to utilize previous sweeps.
   Args:
       sweeps_num (int): Number of sweeps. Defaults to 10.
       load_dim (int): Dimension number of the loaded points. Defaults to 5.
       use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
   """


   def __init__(self,
                use_occ3d=False,
                load_dim=18,
                use_dim=[0, 1, 2, 8, 9, 18],
                sweeps_num=5,
                pc_range=[-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]):
       self.use_occ3d = use_occ3d
       self.load_dim = load_dim
       self.use_dim = use_dim
       self.sweeps_num = sweeps_num
       self.pc_range = pc_range


   def _load_points(self, pts_filename):
       """Private function to load point clouds data.
       Args:
           pts_filename (str): Filename of point clouds data.
       Returns:
           np.ndarray: An array containing point clouds data.
           [N, 18]
       """
       radar_obj = RadarPointCloud.from_file(pts_filename)


       #[18, N]
       points = radar_obj.points


       return points.transpose().astype(np.float32)


   def __call__(self, results):
       """Call function to load multi-sweep point clouds from files.
       Args:
           results (dict): Result dict containing multi-sweep point cloud \
               filenames.
       Returns:
           dict: The result dict containing the multi-sweep points data. \
               Added key and value are described below.
               - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                   cloud arrays.
       """
       radars_dict = results['radars']
       lidar2ego_rotation = np.array(results['lidar_points']['lidar2ego'])[:3,:3]
       lidar2ego_translation = np.array(results['lidar_points']['lidar2ego'])[:3,-1]
       points_sweep_list = []
       for key, sweeps in radars_dict.items():
           if len(sweeps) < self.sweeps_num:
               idxes = list(range(len(sweeps)))
           else:
               idxes = list(range(self.sweeps_num))
          
           ts = sweeps[0]['timestamp'] * 1e-6
           for idx in idxes:
               sweep = sweeps[idx]


               points_sweep = self._load_points(sweep['data_path'])
               points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)


               timestamp = sweep['timestamp'] * 1e-6
               time_diff = ts - timestamp
               time_diff = np.ones((points_sweep.shape[0], 1)) * time_diff


               # velocity compensated by the ego motion in sensor frame
               velo_comp = points_sweep[:, 8:10]
               velo_comp = np.concatenate((velo_comp, np.zeros((velo_comp.shape[0], 1))), 1)
               velo_comp = velo_comp @ sweep['sensor2lidar_rotation'].T
               if self.use_occ3d:
                   velo_comp = velo_comp @ lidar2ego_rotation.T
               velo_comp = velo_comp[:, :2]


               # velocity in sensor frame
               velo = points_sweep[:, 6:8]
               velo = np.concatenate((velo, np.zeros((velo.shape[0], 1))), 1)
               velo = velo @ sweep['sensor2lidar_rotation'].T
               if self.use_occ3d:
                   velo = velo @ lidar2ego_rotation.T
               velo = velo[:, :2]


               points_sweep[:, :3] = points_sweep[:, :3] @ sweep['sensor2lidar_rotation'].T
               points_sweep[:, :3] += sweep['sensor2lidar_translation']
               if self.use_occ3d:
                   points_sweep[:, :3] = points_sweep[:, :3] @ lidar2ego_rotation.T
                   points_sweep[:, :3] += lidar2ego_translation

               points_sweep_ = np.concatenate(
                   [points_sweep[:, :6], velo,
                    velo_comp, points_sweep[:, 10:],
                    time_diff], axis=1)
               points_sweep_list.append(points_sweep_)
      
       points = np.concatenate(points_sweep_list, axis=0)
      
       points = points[:, self.use_dim]
      
       points = torch.from_numpy(points)
      
       results['radars'] = points
       return self.transform(results)


   def transform(self, results):
       radar_pts = results['radars']
       radar_pts_xyz = radar_pts[:,0:3]
       idx = torch.where((radar_pts_xyz[:,0] > self.pc_range[0])
                         & (radar_pts_xyz[:,1] > self.pc_range[1])
                         & (radar_pts_xyz[:,2] > self.pc_range[2])
                         & (radar_pts_xyz[:,0] < self.pc_range[3])
                         & (radar_pts_xyz[:,1] < self.pc_range[4])
                         & (radar_pts_xyz[:,2] < self.pc_range[5]))
       radar_pts = radar_pts[idx]
       results['radars'] = radar_pts
       return results
  
   def __repr__(self):
       """str: Return a string that describes the module."""
       return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'
