#!/bin/bash
date='2023-08-25_13-03-43'
architecture='pv_rcnn_relation'

# Create the first tmux session and run a command
tmux new-session -d -s Session1
tmux send-keys -t Session1 "(cd tools/; python test.py --cfg ./cfgs/kitti_models/$architecture.yaml --eval_all --extra_tag $date)" C-m

sleep 300

# Create the second tmux session and run another command
tmux new-session -d -s Session2
tmux send-keys -t Session2 "(cd output/cfgs/kitti_models/$architecture/$date/eval/eval_all_default/default; tensorboard dev upload --logdir tensorboard_val --name \"${architecture//-/ } KITTI Evaluation $date\")" C-m

