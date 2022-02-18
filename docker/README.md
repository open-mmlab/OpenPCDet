# Guidance to use OpenPCDet with docker

You can either build the docker image through Dockerfile or pull the docker image from dockerhub. Please make sure nvidia-docker is corretly installed.

## Build Through Dockerfile
Build docker image that support OpenPCDet through:
```shell script
docker build ./ -t openpcdet-docker
```
Note that if you would like to use dynamic voxelization, you need further install [`torch_scatter`](https://github.com/rusty1s/pytorch_scatter) package. 

From this Dockerfile, the installed version of spconv is 2.x, if you would like to use spconv 1.2.1, please follow these steps:
```shell script
git clone -b v1.2.1 https://github.com/djiajunustc/spconv spconv --recursive
cd spconv
python setup.py bdist_wheel
cd ./dist
pip install *.whl
```

## Pull From Dockerhub
Run the following script to pull the docker image:
```shell script
docker pull djiajun1206/pcdet:pytorch1.6
```
