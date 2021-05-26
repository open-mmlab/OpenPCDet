# Semi-supervised learning

##  1. Parse the instruction doc of scenes.
tools/generate_pseudo_label/parse_scene_instruction_doc.py
```shell script
python parse_scene_instruction_doc.py
```` 

## 2. Generate pseudo labels with pre-trained pv_rcnn and unlabeled points, and generate training infos and ground truth database.
### A. preparation
 - create data config file, and make sure DATA_PATH is the path for newly generated dataset, which should be the same with "--root_path" in the following setting.
 - create training config file, and make sure DATA_CONFIG:_BASE_CONFIG_ is same with data config file.

### B. run script
#### Specify Params
Meaning of Arguments:
 - --teacher_cfg_file, the config of pretrained teacher network for pseudo labels.
 - --gpus, the gpus that can be used for inference, e.g. 1,2.
 - --infer_bs, the batch size used for inference.
 - -- infer_workers, the workers used for inference.
 - --score_thresh, specify the score thresh for generating pseudo labels.
 - --ckpt, the pretrained model of teacher network.
 - --ext, the extension of your point cloud data file, default='.bin'.
 - --copy_org_data, whether to copy labeled data and raw velodyne. if True, the following 3 parameters need to be specified.
 - --org_velodyne_path, the velodyne path of original labeled data(please not end with "/").
 - --org_label_path, the label path of original labeled data(please not end with "/").
 - --org_raw_data_path, the path of raw velodyne without label(please not end with "/").
 - --root_path, the root path for generated dataset(please not end with "/", and make sure the data root is same with that in cofg_file), e.g. path/to/pseudo_label_dataset.
 - --org_list_file, the orginal list file to create new train list file.
 - --org_val_list_file, the original val list file to add sub folder.
 - --cfg_file, the dataset config file to create infos.
 - --use_npy_file, whether to use offline-agumented npy file.
 - --check_empty_label, whether to check empty labels.
 - --need_count_labels, whether to count anchor size in labels.
 - --train_cfg_file, the yaml file for training.
 - --workers, number of multi thread workers.

#### Use command or run Script

```
# enter tools folder
cd tools
# m1: run command
python semi_supervised_train.py --teacher_cfg_file cfgs/neolix_models/pv_rcnn_1028.yaml --gpus 1,2 --score_thresh 0.25 --ckpt /nfs/neolix_data1/pv_rcnn_1028.pth --cfg_file cfgs/dataset_configs/neolix_pseudo_dataset_9w9_0p1.yaml --train_cfg_file cfgs/neolix_models/pointpillar_1031.yaml --copy_org_data True --org_velodyne_path /nfs/neolix_data1/neolix_dataset/develop_dataset/pseudo_label_dataset/training/velodyne/labeled --org_label_path /nfs/neolix_data1/neolix_dataset/develop_dataset/pseudo_label_dataset/training/label_2/labeled --org_raw_data_path /nfs/neolix_data1/neolix_dataset/develop_dataset/pseudo_label_dataset/training/velodyne/pseudo --root_path /nfs/neolix_data1/neolix_dataset/develop_dataset/pseudo_label_dataset_0p1 --org_list_file /nfs/neolix_data1/neolix_dataset/develop_dataset/lidar_object_detection/ID_1022/ImageSets/train.txt --org_val_list_file /nfs/neolix_data1/neolix_dataset/develop_dataset/lidar_object_detection/ID_1022/ImageSets/val.txt --workers 8
# m2: change params in script and run
bash scripts/generate_pseudo_label_data.sh
```

## 3. Train with labeled and pseudo datasets in multi GPUS.
```shell script
bash scripts/dist_train.sh ${GPU_NUM} --cfg_file ${CONFIG_FILE}
```
