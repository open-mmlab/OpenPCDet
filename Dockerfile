# This tag does not exist anymore it was changed to the next closest: FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu18.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Install basics
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
RUN apt-get update -y \
    && apt-get install -y build-essential \
    && apt-get install -y apt-utils git curl ca-certificates bzip2 tree htop wget \
    && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev bmon iotop g++ python3.8 python3.8-dev python3.8-distutils

# Install cmake v3.13.2
RUN apt-get purge -y cmake && \
    mkdir /root/temp && \
    cd /root/temp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.13.2/cmake-3.13.2.tar.gz && \
    tar -xzvf cmake-3.13.2.tar.gz && \
    cd cmake-3.13.2 && \
    bash ./bootstrap && \
    make && \
    make install && \
    cmake --version && \
    rm -rf /root/temp

# Install python
RUN ln -sv /usr/bin/python3.8 /usr/bin/python
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python get-pip.py && \
	rm get-pip.py

# Install python packages
RUN PIP_INSTALL="python -m pip --no-cache-dir install" && \
    # $PIP_INSTALL numpy==1.19.3 llvmlite numba
    $PIP_INSTALL numpy==1.24.4 llvmlite numba

# Install torch and torchvision
# See https://pytorch.org/ for other options if you use a different version of CUDA
# RUN pip install --user torch==1.6 torchvision==0.7.0 -f https://download.pytorch.org/whl/cu102/torch_stable.html
# RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install torch torchvision torchaudio

# Install python packages
RUN PIP_INSTALL="python -m pip --no-cache-dir install" && \
    $PIP_INSTALL tensorboardX easydict pyyaml scikit-image tqdm SharedArray six

WORKDIR /root

# Install Boost geometry
RUN wget https://jaist.dl.sourceforge.net/project/boost/boost/1.68.0/boost_1_68_0.tar.gz && \
    tar xzvf boost_1_68_0.tar.gz && \
    cp -r ./boost_1_68_0/boost /usr/include && \
    rm -rf ./boost_1_68_0 && \
    rm -rf ./boost_1_68_0.tar.gz

# A weired problem that hasn't been solved yet
RUN pip uninstall -y SharedArray && \
    pip install SharedArray

RUN pip install spconv-cu117

RUN pip install torch_geometric

RUN pip install open3d

RUN pip install mayavi

RUN pip install kornia

RUN apt-get update && apt-get install -y libgl1-mesa-glx

RUN apt-get update && apt-get install -y \
    openssh-client \
    git

RUN git clone https://github.com/MarcBrede/OpenPCDet.git

ENV PYTHONPATH /root/OpenPCDet/

RUN pip install av2

RUN pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-2.0.1+cu117.html
RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.0.1+cu117.html
RUN pip install tensorboard

# RUN ls /root/OpenPCDet/

# RUN python /root/OpenPCDet/setup.py develop