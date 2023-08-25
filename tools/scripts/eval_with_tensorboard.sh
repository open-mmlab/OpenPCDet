date='put date here'
architecture='pv_rcnn_relation'

tmux send-keys -t 0 "python test.py --cfg ./cfgs/kitti_models/$architecture.yaml --eval_all --extra_tag $date" C-m

sleep 2

tmux send-keys -t 1 "(cd output/cfgs/kitti_models/$architecture/$date/eval/eval_all_default/default; tensorboard dev upload --logdir tensorboard_val --name \"${architecture//-/ } KITTI Evaluation $date\")" C-m