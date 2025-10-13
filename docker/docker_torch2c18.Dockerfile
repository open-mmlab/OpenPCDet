```
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04


ENV PYTHONUNBUFFERED=1 
ENV FORCE_CUDA="1"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# SYSTEM
RUN apt-get update --yes --quiet && DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    software-properties-common \
    build-essential apt-utils \
    wget curl vim git ca-certificates kmod \
 && rm -rf /var/lib/apt/lists/*
RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6  -y
# PYTHON 3.10
RUN add-apt-repository --yes ppa:deadsnakes/ppa 
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test
RUN apt install -y g++-11
RUN apt install -y gcc-11
RUN apt install -y libsparsehash-dev
RUN apt-get update --yes --quiet
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3.10-lib2to3 \
    python3.10-gdbm \
    python3.10-tk \
    pip

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 999 \
    && update-alternatives --config python3 && ln -s /usr/bin/python3 /usr/bin/python

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10



RUN pip3 install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118



RUN pip3 install opencv-python==4.7.0.68
RUN pip3 install llvmlite numba tensorboardX easydict pyyaml scikit-image tqdm SharedArray open3d mayavi av2 kornia==0.6.5 pyquaternion
RUN pip3 install spconv-cu118

# remember you pull the whole repo before that and run docker build ... from OpenPCDet, not Docker folder or tools
WORKDIR /usr/local/
COPY . OpenPCDet
WORKDIR /usr/local/OpenPCDet
RUN pip3 install -r requirements.txt

# next, run docker and
# python3 setup.py develop
```
