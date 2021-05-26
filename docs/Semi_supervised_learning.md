# Semi-supervised learning


## 1. 数据采集阶段
录制records，利用解析工具将原始records解析为bin文件，存储于存储服务器。

## 2. 伪标签数据集生成阶段
将步骤1中的原始lidar数据(bin文件)转换为训练需要格式。

### 准备：
 - 创建数据集配置文件，存放于OpenPCDet/tools/cfgs/dataset_configs/，配置DATA_PATH为用来存放伪标签数据集的路径，需要与运行脚本中的root_path保持一致；
 - 创建训练配置文件，存放于OpenPCDet/tools/cfgs/neolix_models/，确保DATA_CONFIG:_BASE_CONFIG指向了上一步的数据集配置文件；
### 运行：
 - 进入tools目录，命令行直接运行  
```
    python semi_supervised_train.py --teacher_cfg_file cfgs/neolix_models/pv_rcnn_1028.yaml --gpus 1,2 --score_thresh 0.25 --ckpt /nfs/neolix_data1/pv_rcnn_1028.pth --cfg_file cfgs/dataset_configs/neolix_pseudo_dataset_9w9_0p1.yaml --train_cfg_file cfgs/neolix_models/pointpillar_1031.yaml --copy_org_data True --org_velodyne_path /nfs/neolix_data1/neolix_dataset/develop_dataset/pseudo_label_dataset/training/velodyne/labeled --org_label_path /nfs/neolix_data1/neolix_dataset/develop_dataset/pseudo_label_dataset/training/label_2/labeled --org_raw_data_path /nfs/neolix_data1/neolix_dataset/develop_dataset/pseudo_label_dataset/training/velodyne/pseudo --root_path /nfs/neolix_data1/neolix_dataset/develop_dataset/pseudo_label_dataset_0p1 --org_list_file /nfs/neolix_data1/neolix_dataset/develop_dataset/lidar_object_detection/ID_1022/ImageSets/train.txt --org_val_list_file /nfs/neolix_data1/neolix_dataset/develop_dataset/lidar_object_detection/ID_1022/ImageSets/val.txt --workers 8
```
 - 修改sh脚本配置并在tools目录下运行  
```
    bash scripts/generate_pseudo_label_data.sh
```

 - 配置参数的含义(建议使用前对加粗参数进行修改确认)

   - teacher_cfg_file，预训练的教师网络配置文件

   - gpus，指定可用的gpus，如1,2（需要注意的是，仅支持单卡推理，即使用指定的第一个gpu进行推理），默认为0卡

   - infer_bs，指定推理的batch_size，默认为12

   - infer_workers，指定推理时数据读取的worker数，默认为4

   - score_thresh，指定生成伪标签使用的置信度阈值

   - ckpt，预训练教师模型存放位置

   - ext，待生成为标签的点云文件的后缀，默认为bin

   - copy_org_data, 是否需要拷贝已标注数据和未标注的原始点云到新目录

   - org_velodyne_path，已标注数据点云路径，不要以"/"结尾

   - org_label_path，已标注数据标签路径，不要以"/"结尾

   - org_raw_data_path，未标注数据存放位置，即步骤1中存放的路径，不要以"/"结尾

   - root_path，存放伪标签数据集的路径，不要以"/"结尾

   - org_list_file，已标注数据训练集列表文件路径

   - org_val_list_file，已标注数据验证集的列表文件路径

   - cfg_file，数据集配置文件（即上一步准备过程中创建的数据集配置文件）

   - use_npy_file，是否需要生成离线预处理及增广后的npy文件，默认不需要

   - check_empty_label，是否需要检查空标签，若生成的时候已经过滤，则无需检查
   - need_count_labels，是否需要统计anchor尺寸信息并更新配置文件，默认需要

   - train_cfg_file，指定用于训练的配置文件（即准备过程中创建的训练配置文件）

   - workers，指定多进程的进程数，默认为4

## 3. 模型训练阶段
基于步骤2中生成的数据集配置文件和训练配置文件进行训练。
```shell script
bash scripts/dist_train.sh ${GPU_NUM} --cfg_file ${CONFIG_FILE}
```
