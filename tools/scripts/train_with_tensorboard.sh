#!/bin/bash

# Getting the current date
date=$(date "+%Y-%m-%d_%H-%M-%S")

# Defining the architecture variable
architecture="pv_rcnn_relation"

# Sending the training command to tmux session 0
tmux send-keys -t 0 "python train.py --cfg_file ./cfgs/kitti_models/$architecture.yaml --extra_tag $date" C-m

# Giving the command time to start
sleep 2

# Navigating to the output directory and starting tensorboard in tmux session 1
# Note: the architecture name is converted to uppercase with "${architecture^^}"
tmux send-keys -t 1 "(cd output/cfgs/kitti_models/$architecture/default/; tensorboard dev upload --logdir tensorboard --name \"${architecture^^} KITTI $date\")" C-m

