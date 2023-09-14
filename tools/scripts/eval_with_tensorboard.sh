#!/bin/bash
date='2023-09-14_13-12-43'
# architecture='pv_rcnn_relation'
# architecture='centerpoint'
# architecture='pv_rcnn_plusplus_reproduced_by_community'
# architecture='pv_rcnn_plusplus_relation'
architecture='pv_rcnn_relation_fc'

# tmux kill-server

# Create the first tmux session and run a command
tmux new-session -d -s Session3
tmux send-keys -t Session3 "(cd tools/; python test.py --cfg ./cfgs/kitti_models/$architecture.yaml --eval_all --extra_tag $date --max_waiting_mins 1440)" C-m

sleep 300

# Create the second tmux session and run another command
tmux new-session -d -s Session4
tmux send-keys -t Session4 "(cd output/cfgs/kitti_models/$architecture/$date/eval/eval_all_default/default; tensorboard dev upload --logdir tensorboard_val --name "${architecture//-/ }_KITTI_Evaluation_$date")" C-m
