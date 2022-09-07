#!/bin/bash

#Modify these paths if required
export NUSCENES_MINI_PATH=/root/shared_data/nuscenes/nuscenes-mini/
export KITTI_PATH=/root/shared_data/kitti
export MODELS_PATH=/root/shared_data/models
export SPLITS_FILE=/root/nuscenes-devkit/python-sdk/nuscenes/utils/splits.py
export NSYS_PATH=/opt/nvidia/nsight-systems/2022.3.3/target-linux-tegra-armv8/nsys

mkdir -p ../data/nuscenes
pushd ../data/nuscenes
ln -s $NUSCENES_MINI_PATH v1.0-mini
popd


pushd ../data/kitti
ln -s $KITTI_PATH/training
ln -s $KITTI_PATH/testing
for m in $KITTI_PATH/generated_data/*; do ln -s $m; done
popd

pushd ..
ln -s $MODELS_PATH
popd

patch $SPLITS_FILE < splits.patch

pushd /usr/bin
ln -s $NSYS_PATH
popd
