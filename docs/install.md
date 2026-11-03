# Step-by-step installation instructions

**a. Create a conda virtual environment and activate it.**
```shell
conda create -n TFusionOcc python=3.8 -y
conda activate TFusionOcc
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

**c. Install mmengine, mmcv, mmdet, mmdet3d, and mmseg.**
```shell
pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc4'
mim install 'mmdet>=3.0.0'
mim install "mmdet3d>=1.1.0"
pip install "mmsegmentation>=1.0.0"
```
**d. Install others.**
```shell
cd ./tfusion/localagg_T_prob_${all T-prob packages}$ && pip install -e .
cd ./DFA3D && bash setup.sh 0
cd ./lib/pointops && pip install -e .
```
**e. Download code and backbone pretrain weight.**
```shell
git clone https://github.com/DanielMing123/TFusionOcc.git
cd TFusionOcc
mkdir ckpt
wget https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth
```
**f. The TFusionOcc repo core structure should be like the following**
```
TFusionOcc
├── ckpt/
├── configs/
├── DFA3D/
├── lib/
├── tfusion/
├── tools/
├── data/
│   ├── nuscenes/
│   ├── nuScenes-c-image/
│   ├── nuScenes-c-lidar/
```
