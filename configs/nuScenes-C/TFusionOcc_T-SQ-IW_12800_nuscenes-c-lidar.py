import math
_base_ = ['../_base_/default_runtime.py']
custom_imports = dict(imports=['tfusion'], allow_failed_imports=False)

load_from = 'ckpt/T_SQ_IW_12800_SOTA.pth'

dataset_type = 'NuScenesSegDataset'
data_root = 'data/nuscenes'
data_prefix = dict(
    pts='samples/LIDAR_TOP',
    pts_semantic_mask='lidarseg/v1.0-trainval',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT')

input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None

pc_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
pol_range= [0.0, -math.pi, -5.0, 50.0*math.sqrt(2), math.pi, 3.0]
dbound = [0.0, 71.0, 0.5] # 71/58, 0.25/0.5
grid_size_pol = [240, 360, 16]   # 480, 360, 32 Key-Factor
grid_size_occ= [200, 200, 16]
use_sq=False
use_sq_iw=True
use_occ3d=False
find_unused_parameters=False

model = dict(
    type='TFusionOcc',
    use_occ3d=use_occ3d,
    use_grid_mask=True,
    data_preprocessor=dict(
        type='TFusionDataPreprocessor',
        pad_size_divisor=32,
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0]), 
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),  
        stage_with_dcn=(False, False, True, True)),
    second_neck=dict(
        type='SECONDFPN',
        in_channels=[512, 1024, 2048],
        upsample_strides=[1, 2, 4],
        out_channels=[128, # dim * 3 = final_dim
                      128, 
                      128]), 
    neck=dict(
        type='mmdet.FPN',
        in_channels=[512, 1024, 2048],
        out_channels=128,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=3,
        relu_before_extra_convs=True),
    cylindrical_partition = dict(
        type='Cylindrical_Partition',
        num_lidar_anchor=9600,  # 9600/19200
        num_cam_anchor=3200,    # 3200/6400
        k=5, # 5/2
        grid_size_pol=grid_size_pol,
        grid_size_occ=grid_size_occ, 
        in_channels=11, 
        out_channels=128,  # 128
        dbound=dbound, # lidar align with camera
        pc_range=pc_range,
        pol_range=pol_range,
        lidar_encoder=dict(
            type='Cylinder',
            grid_size=grid_size_pol,
            input_channels=128,
            base_channels=32)),
    transformer = dict(
        type='Transformer',
        num_layers=4, # 6 4 3 Key-Factor
        num_head=4,
        num_levels=3,
        num_cams=6,
        embed_dims=128,  # 128
        dbound=dbound, # lidar align with camera
        grid_size_pol=grid_size_pol, 
        grid_size_occ=grid_size_occ,
        pc_range=pc_range,
        pol_range=pol_range,
        kernel_size=3, # 3/5 
        scale_range=[0.01, 2.5], # 2.5/1.8
        num_anchor=12800, # 12800/25600
        splatting_head=dict(
            type='SplattingHead',
            scale_multiplier=5, # 5/4
            H=200, 
            W=200, 
            D=16,  
            pc_min=[-50.0, -50.0, -5.0],
            grid_size=0.5,
            use_sq_iw=use_sq_iw),
        use_sq_iw=use_sq_iw)) 

train_pipeline = [
    dict(type='BEVLoadMultiViewImageFromFiles',
        to_float32=False,
        color_type='unchanged',
        num_views=6,
        backend_args=backend_args),
    dict(type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(type='LoadPointsFromMultiSweeps',
        sweeps_num=10, 
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(type='PointsRangeFilter', point_cloud_range=pc_range),
    dict(type='LoadOccupancy'),
    dict(type='LoadDepthGT'),
    dict(type='Cart2Polar',
         grid_size_pol=grid_size_pol,
         grid_size_occ=grid_size_occ,
         voxel_size=[0.5, 0.5, 0.5],
         pc_range=pc_range,
         pol_range=pol_range),
    dict(type='MultiViewWrapper',
        transforms=dict(type='PhotoMetricDistortion3D')),
    dict(type='Custom3DPack',
        keys=['img', 'points','occ_200','depth_gt','pts_pol_vox_coords','pts_pol_feat','sample_loc'], 
        meta_keys=['lidar2img'])
]

corruption_root='data/nuScenes-c-lidar/'
corruption='beam_missing' # ['beam_missing', 'cross_sensor', 'crosstalk', 'incomplete_echo', 'motion_blur', 'wet_ground', 'fog', 'snow', 'Clean'] 'Clean' means no-corruption
severity='light' # ['light', 'moderate', 'heavy']
val_pipeline = [
    dict(type='BEVLoadMultiViewImageFromFiles',
        to_float32=False,
        color_type='unchanged',
        num_views=6,
        backend_args=backend_args),
    dict(type='Custom_LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args,
        corruption=corruption,
        severity=severity,
        corruption_root=corruption_root),
    dict(type='LoadPointsFromMultiSweeps',
        sweeps_num=0, # 10
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(type='PointsRangeFilter', point_cloud_range=pc_range),
    dict(type='LoadOccupancy'),
    dict(type='LoadDepthGT'),
    dict(type='Cart2Polar',
         grid_size_pol=grid_size_pol,
         grid_size_occ=grid_size_occ,
         voxel_size=[0.5, 0.5, 0.5],
         pc_range=pc_range,
         pol_range=pol_range), 
    dict(type='Custom3DPack',
        keys=['img', 'points','occ_200','depth_gt','pts_pol_vox_coords','pts_pol_feat','sample_loc'], 
        meta_keys=['lidar2img'])
]

test_pipeline = val_pipeline



train_dataloader = dict(
    batch_size=1, 
    num_workers=4,
    persistent_workers=True,
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=True),  # shuffle=True
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=data_prefix,
        ann_file='nuscenes_infos_occfusion_train.pkl', #'nuscenes_infos_occfusion_train.pkl', 'nuscenes_infos_occfusion_val_1.pkl'
        pipeline=train_pipeline,
        test_mode=False))

val_dataloader = dict(
    batch_size=4, 
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=data_prefix,
        ann_file='nuscenes_infos_occfusion_val.pkl',
        pipeline=val_pipeline,
        test_mode=True)) 

test_dataloader = val_dataloader

val_evaluator = dict(type='EvalMetric')

test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-5, weight_decay=0.01), 
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.1),
    }),
    clip_grad=dict(max_norm=35, norm_type=2),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-5, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=24,
        by_epoch=True,
        eta_min=1e-6,
        convert_to_iter_based=True)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=10, val_begin=11, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1))