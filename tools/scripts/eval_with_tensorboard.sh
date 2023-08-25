date='2023-08-24_13-25-38'
architecture='pv_rcnn_relation'

tmux send-keys -t 0 "(cd tools/; python test.py --cfg ./cfgs/kitti_models/$architecture.yaml --eval_all --extra_tag $date)" C-m

tmux new-window -n tensorboard
tmux send-keys -t 1 "(cd output/cfgs/kitti_models/$architecture/$date/eval/eval_all_default/default; tensorboard dev upload --logdir tensorboard_val --name \"${architecture//-/ } KITTI Evaluation $date\")" C-m
