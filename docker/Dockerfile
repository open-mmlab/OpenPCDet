FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu18.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Install basics
RUN apt-get update -y \
    && apt-get install build-essential \
    && apt-get install -y apt-utils git curl ca-certificates bzip2 tree htop wget \
    && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev bmon iotop g++ python3.6 python3.6-dev python3.6-distutils

RUN apt-get update && apt-get install -y software-properties-common
RUN apt-get update -q && \
    apt-get install -yq wget curl git build-essential vim sudo lsb-release locales bash-completion

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
RUN ln -sv /usr/bin/python3.6 /usr/bin/python
# RUN wget https://bootstrap.pypa.io/get-pip.py && \
RUN wget https://bootstrap.pypa.io/pip/3.6/get-pip.py && \
	python get-pip.py && \
	rm get-pip.py

# Install python packages
RUN PIP_INSTALL="python -m pip --no-cache-dir install" && \
    $PIP_INSTALL numpy==1.19.3 llvmlite numba 

# Install torch and torchvision
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip3 install numpy==1.19.5 #to solve numpy dependency issues with opencv!

# Install python packages
RUN PIP_INSTALL="python -m pip --no-cache-dir install" && \
    $PIP_INSTALL tensorboardX easydict pyyaml scikit-image tqdm SharedArray six

# WORKDIR /root

# Install Boost geometry
RUN wget https://jaist.dl.sourceforge.net/project/boost/boost/1.68.0/boost_1_68_0.tar.gz && \
    tar xzvf boost_1_68_0.tar.gz && \
    cp -r ./boost_1_68_0/boost /usr/include && \
    rm -rf ./boost_1_68_0 && \
    rm -rf ./boost_1_68_0.tar.gz 

# A weired problem that hasn't been solved yet
RUN pip uninstall -y SharedArray && \
    pip install SharedArray

RUN pip install spconv-cu113

RUN pip install open3d==0.13
RUN apt install -y libgl1-mesa-glx

#just for running the Demo we need to add:
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python

#to get a GUI one needs to install:
RUN apt-get install -y python3-tk 
RUN apt-get install -y libnvidia-gl-510 x11-apps #this needs to match your GPU driver version!!!!!!!!!!!!!!

RUN apt-get install -y tmux tmuxp vim 
RUN mkdir -p /OpenPCDet/data/kitti/ImageSets
RUN mkdir -p /OpenPCDet/data/kitti/testing
RUN mkdir -p /OpenPCDet/data/kitti/training
RUN mkdir /OpenPCDet/checkpoints

#ROS
RUN apt-get install -y net-tools iputils-ping
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu bionic main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -k https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo apt-key add -

RUN apt-get update -q && \
    apt-get install -y ros-melodic-desktop-full python-rosdep &&\
    # apt-get install -y ros-melodic-desktop-full python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential && \
    rm -rf /var/lib/#apt/lists/*
RUN rosdep init
RUN rosdep update
RUN apt install -y python3-pip python3-all-dev python3-rospkg
RUN apt install -y ros-melodic-desktop-full --fix-missing

# RUN rm /usr/bin/python
# RUN rm /usr/bin/python3
# RUN ln -sv /usr/bin/python3.6 /usr/bin/python

RUN locale-gen en_US.UTF-8
RUN useradd -m -d /home/ubuntu ubuntu -p `perl -e 'print crypt("ubuntu", "salt"),"\n"'` && \
    echo "ubuntu ALL=(ALL) ALL" >> /etc/sudoers
USER ubuntu
# WORKDIR /home/ubuntu
ENV HOME=/home/ubuntu \
    CATKIN_SHELL=bash
ENV LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8

RUN mkdir -p ~/catkin_ws/src \
    && /bin/bash -c '. /opt/ros/melodic/setup.bash; catkin_init_workspace $HOME/catkin_ws/src' \
    && /bin/bash -c '. /opt/ros/melodic/setup.bash; cd $HOME/catkin_ws; catkin_make'
RUN mkdir -p ~/catkin_ws/build
RUN mkdir -p ~/catkin_ws/devel

RUN echo 'source /opt/ros/melodic/setup.bash' >> ~/.bashrc \
    && echo 'source ~/catkin_ws/devel/setup.bash' >> ~/.bashrc
# COPY ./docker/ros_entrypoint.sh /
# ENTRYPOINT ["/ros_entrypoint.sh"]
# CMD ["/bin/bash"]
##ROS

COPY pcdet /OpenPCDet/pcdet
#COPY tools /OpenPCDet/tools
COPY docs /OpenPCDet/docs

ENV PYTHONPATH="/OpenPCDet/:$PYTHONPATH"
ADD setup.py /OpenPCDet/

WORKDIR /OpenPCDet
#RUN python /OpenPCDet/setup.py develop


