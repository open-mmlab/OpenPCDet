#!/bin/bash
# date=$(date "+%Y-%m-%d_%H-%M-%S")
date=2023-10-24_09-25-41
# date='2023-10-19_12-02-51'
# architecture='pv_rcnn'
architecture="pv_rcnn_relation"
# architecture='centerpoint'
# architecture='pv_rcnn_plusplus_reproduced_by_community'
# architecture='centerpoint_twostage'
# architecture='pv_rcnn_plusplus_relation'
# architecture='pv_rcnn_relation_fc'
# architecture='pv_rcnn_car_class_only'
# architecture='pv_rcnn_relation_car_class_only'
# architecture='pv_rcnn_plusplus_reproduced_by_community_car_class_only'
# architecture='pv_rcnn_plusplus_relation_car_class_only'
# architecture='PartA2_car_class_only'
# architecture='PartA2_relation_car_class_only'
# architecture='voxel_rcnn_car'
# architecture='voxel_rcnn_relation_car'
# architecture='pv_rcnn_frozen'
# architecture='pv_rcnn_frozen_relation'
# architecture='pv_rcnn_BADet_car_class_only'

# data='kitti'
data='waymo'

# tmux kill-server

# Sending the training command to tmux session 0
# tmux new-session -d -s Session1
# tmux send-keys -t Session1 "(cd tools/; python train.py --cfg_file ./cfgs/${data}_models/$architecture.yaml --extra_tag $date)" C-m

tmux new-session -d -s Session3
tmux send-keys -t Session3 "(cd tools/; python test.py --cfg ./cfgs/${data}_models/$architecture.yaml --eval_all --extra_tag $date --max_waiting_mins 1440)" C-m

sleep 100

# tmux new-session -d -s Session2
# tmux send-keys -t Session2 "(cd output/cfgs/${data}_models/$architecture/$date/; tensorboard dev upload --logdir tensorboard --name \"${architecture^^} $data $date\")" C-m


# Create the second tmux session and run another command
tmux new-session -d -s Session4
tmux send-keys -t Session4 "(cd output/cfgs/${data}_models/$architecture/$date/eval/eval_all_default/default; tensorboard dev upload --logdir tensorboard_val --name "${architecture//-/ }_$data_Evaluation_$date")" C-m

# evaluate the model for one epoch

# tmux new-session -d -s Session3
# tmux send-keys -t Session3 "(cd tools/; python test.py --cfg_file ./cfgs/${data}_models/${architecture}.yaml --extra_tag $date --ckpt ../output/cfgs/${data}_models/${architecture}/${date}/ckpt/checkpoint_epoch_30.pth)" C-m
