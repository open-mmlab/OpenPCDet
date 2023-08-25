#!/bin/bash

# Getting the current date
date=$(date "+%Y-%m-%d_%H-%M-%S")

# Defining the architecture variable
architecture="pv_rcnn_relation"

# Sending the training command to tmux session 0
tmux send-keys -t 0 "(cd tools/; python train.py --cfg_file ./cfgs/kitti_models/$architecture.yaml --extra_tag $date)" C-m


tmux new-window -n tensorboard
tmux send-keys -t 1 "(cd output/cfgs/kitti_models/$architecture/default/; tensorboard dev upload --logdir tensorboard --name \"${architecture^^} KITTI $date\")" C-m

