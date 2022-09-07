#!/bin/bash

#Modify these paths if required
export NUSCENES_MINI_PATH=/root/shared_data/nuscenes/nuscenes-mini/
export MODELS_PATH=/root/shared_data/models
export SPLITS_FILE=/root/nuscenes-devkit/python-sdk/nuscenes/utils/splits.py
export NSYS_PATH=/opt/nvidia/nsight-systems/2022.3.3/target-linux-tegra-armv8/nsys

git pull

mkdir -p ../data/nuscenes
pushd ../data/nuscenes
ln -s $NUSCENES_MINI_PATH v1.0-mini
popd

mkdir -p ../output/nuscenes_models
pushd ../output/nuscenes_models
for m in $MODELS_PATH/*; do ln -s $m; done
popd

patch $SPLITS_FILE < splits.patch

pushd /usr/bin
ln -s $NSYS_PATH
popd
