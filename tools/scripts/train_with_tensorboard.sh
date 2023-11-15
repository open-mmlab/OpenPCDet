#!/bin/bash
date=$(date "+%Y-%m-%d_%H-%M-%S")
architecture="pv_rcnn_relation"

# tmux kill-server

# Sending the training command to tmux session 0
tmux new-session -d -s Session1
tmux send-keys -t Session1 "(cd tools/; python train.py --cfg_file ./cfgs/kitti_models/$architecture.yaml --extra_tag $date)" C-m

sleep 300

tmux new-session -d -s Session2
tmux send-keys -t Session2 "(cd output/cfgs/kitti_models/$architecture/$date/; tensorboard dev upload --logdir tensorboard --name \"${architecture^^} KITTI $date\")" C-m

