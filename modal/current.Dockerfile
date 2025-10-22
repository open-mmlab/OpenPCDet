FROM nvidia/cuda:11.6.2-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

# Get all dependencies
RUN apt-get update && apt-get install -y \
    git zip build-essential cmake libssl-dev python3-dev  python3-pip python3-pip cmake ninja-build git wget ca-certificates ffmpeg libsm6 libxext6 &&\
    rm -rf /var/lib/apt/lists/*

RUN ln -sv /usr/bin/python3 /usr/bin/python
ENV PATH="/root/.local/bin:${PATH}"
RUN wget -O /root/get-pip.py https://bootstrap.pypa.io/get-pip.py && python3 /root/get-pip.py --user

# PyTorch for CUDA 11.6
RUN pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
ENV TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX;Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing;8.6"

RUN pip install numpy==1.23.0 llvmlite numba opencv-python tensorboardX easydict pyyaml scikit-image tqdm SharedArray open3d mayavi av2 pyquaternion kornia==0.6.8 nuscenes-devkit==1.0.5 spconv-cu116
RUN python -m pip install --user jupyter
RUN git clone https://github.com/open-mmlab/OpenPCDet.git
WORKDIR OpenPCDet
RUN python setup.py develop
RUN pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.13.1+cu116.html

