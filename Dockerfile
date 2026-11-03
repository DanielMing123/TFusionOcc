FROM determinedai/environments-dev:cuda-11.8-pytorch-2.0-gpu-mpi-fb64b20

WORKDIR /workdir

ARG TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0+PTX"
ENV FORCE_CUDA="1"
ENV MMCV_WITH_OPS=1
ENV TZ=Australia/Sydney \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y ffmpeg git htop vim wget unzip sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV, MMDetection and MMSegmentation
RUN pip install spconv-cu118
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
RUN pip install pykitti
RUN pip install timm
RUN pip install debugpy
RUN pip install Cython
RUN pip install -U openmim
RUN pip install mmengine
# RUN pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
RUN pip install mmcv==2.0.1
RUN pip install mmdet==3.0.0rc5
RUN pip install mmdet3d==1.3.0
RUN pip install "mmsegmentation>=1.0.0"
RUN pip install focal_loss_torch
