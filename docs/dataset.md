## NuScenes
Download nuScenes V1.0 full dataset data [HERE](https://www.nuscenes.org/download).

Download nuscenes_infos_occfusion_train.pkl [HERE](https://drive.google.com/file/d/1N-Lu72HDr4LHOs7HXiUpOk8xyq6bJOyb/view?usp=sharing). 

Download nuscenes_infos_occfusion_val.pkl [HERE](https://drive.google.com/file/d/1hf40D53Bj6JZF2jMEeCbDfhvM4aXqvc1/view?usp=sharing).  

Download nuscenes_infos_occfusion_val_rainy.pkl [HERE](https://drive.google.com/file/d/1MZgl6wcZ0_qxS8q6DbkwvVEDixmeodfc/view?usp=sharing).

Download nuscenes_infos_occfusion_val_night.pkl [HERE](https://drive.google.com/file/d/1Pu8oFRp2T2rL96Vf14gDBjQBlLtADQ8e/view?usp=sharing).

## SurroundOcc-NuScenes, Occ3D-NuScenes and NuScenes-C Occupancy Benchmark 
Download the SurroundOcc's 3D Occupancy annotations [HERE](https://github.com/weiyithu/SurroundOcc/blob/main/docs/data.md)
Download the Occ3D's 3D Occupancy annotations [HERE](https://tsinghua-mars-lab.github.io/Occ3D/)
Download the NuScenes-C's 3D Occupancy annotations [HERE](https://github.com/worldbench/RoboBEV)
- only support training and validation set

**dataset structure**
```
TFusionOcc
├── data/
│   ├── nuScenes-c-image/
│   │   ├── Brightness/
│   │   ├── CameraCrash/
│   │   ├── ColorQuant/
│   │   ├── Fog/
│   │   ├── FrameLost/
│   │   ├── LowLight/
│   │   ├── MotionBlur/
│   │   ├── Snow/
│   ├── nuScenes-c-lidar/
│   │   ├── beam_missing/
│   │   ├── cross_sensor/
│   │   ├── crosstalk/
│   │   ├── fog/
│   │   ├── incomplete_echo/
│   │   ├── motion_blur/
│   │   ├── snow/
│   │   ├── wet_ground/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── occ_samples/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── nuscenes_infos_occfusion_train.pkl
|   |   ├── nuscenes_infos_occfusion_val.pkl
|   |   ├── nuscenes_infos_occfusion_val_rainy.pkl
|   |   ├── nuscenes_infos_occfusion_val_night.pkl
```
