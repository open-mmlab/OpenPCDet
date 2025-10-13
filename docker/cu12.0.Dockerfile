FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

# Set environment variables
ENV NVENCODE_CFLAGS="-I/usr/local/cuda/include"
ENV CV_VERSION=4.x
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"

# Get all dependencies
RUN apt-get update && apt-get install -y \
    git zip unzip build-essential python3 cmake python3-pip ffmpeg libsm6 libxext6

WORKDIR /

RUN git clone https://github.com/open-mmlab/OpenPCDet.git

WORKDIR OpenPCDet

RUN pip3 install -r requirements.txt

RUN pip3 install numpy\<2 av2 kornia==0.5.8 spconv-cu120 # waymo-open-dataset-tf-2-11-0

RUN python3 setup.py develop

WORKDIR tools