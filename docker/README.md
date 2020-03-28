To use PCDet in docker please make sure you have `nvidia-docker` installed.

The following steps are required:

1) Build the docker image: `./build.sh`

2) Edit the `run.sh` script to mount KITTI, pretrained models, data exchange as volumes into the docker container.

3) Create a container: `./run.sh`
