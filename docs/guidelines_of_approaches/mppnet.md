## NOTE
**If you want to quickly develop your own model based on MPPNet, our recommended setting is to use mppnet_4frames.yaml, disable `USE_ROI_AUG` and `USE_TRAJ_AUG` flags in the yaml and train 3 epoch. A reference time cost for this setting is about 5 hours, using 8 A100 GPUs.  After finishing your development, you can get stable gains when using mppnet_16frames.yaml, enabling `USE_ROI_AUG` and `USE_TRAJ_AUG` flags and training 6 epoch.**

## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for the installation of `OpenPCDet`.

## Data Preparation
Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) to process the Waymo Open Dataset.

## Training

1.  Train the RPN model for MPPNet (centerpoint_4frames is employed in the paper)
```shell
bash scripts/dist_train.sh ${NUM_GPUS} --cfg_file cfgs/waymo_models/centerpoint_4frames.yaml
```
The ckpt will be saved in ../output/waymo_models/centerpoint_4frames/default/ckpt.

2.  Save the RPN model's prediction results of training and val dataset
```shell
# training
bash scripts/dist_test.sh ${NUM_GPUS}  --cfg_file cfgs/waymo_models/centerpoint_4frames.yaml \
--ckpt ../output/waymo_models/centerpoint_4frames/default/ckpt/checkpoint_epoch_36.pth \
--set DATA_CONFIG.DATA_SPLIT.test train
# val
bash scripts/dist_test.sh ${NUM_GPUS}  --cfg_file cfgs/waymo_models/centerpoint_4frames.yaml \
--ckpt ../output/waymo_models/centerpoint_4frames/default/ckpt/checkpoint_epoch_36.pth \
--set DATA_CONFIG.DATA_SPLIT.test val
```
The prediction results of train and val dataset will be saved in \
../output/waymo_models/centerpoint_4frames/default/eval/epoch_36/train/default/result.pkl,
../output/waymo_models/centerpoint_4frames/default/eval/epoch_36/val/default/result.pkl.

3.  Train MPPNet (using mppnet_4frames as an example)
```shell
bash scripts/dist_train.sh ${NUM_GPUS} --cfg_file cfgs/waymo_models/mppnet_4frames.yaml --batch_size  2  \
--set DATA_CONFIG.ROI_BOXES_PATH.train  ../output/waymo_models/centerpoint_4frames/default/eval/epoch_36/train/default/result.pkl \
 DATA_CONFIG.ROI_BOXES_PATH.test  ../output/waymo_models/centerpoint_4frames/default/eval/epoch_36/val/default/result.pkl
```
When using 16-frame, we can just change the `cfg_file` to mpppnet_16frames.yaml and the `DATA_CONFIG.ROI_BOXES_PATH` is same with 4-frame.\
We can also save the paths of train and val results to ROI_BOXES_PATH in mppnet_4frames.yaml and mppnet_16frames.yaml to avoid using the `set` flag.\
For each GPU, BATCH_SIZE should be at least equal to 2.  When using 16-frame, the reference GPU memory consumption is 29G with BATCH_SIZE=2.\
**Note**: Disable the `USE_ROI_AUG` and `USE_TRAJ_AUG` flag in config yaml can double the training speed with a performance loss of about 0.4%. 

## Evaluation
* Test with a pretrained model:
```shell
# Single GPU
python test.py --cfg_file cfgs/waymo_models/mppnet_4frames.yaml  --batch_size  1 \
--ckpt  ../output/waymo_models/mppnet_4frames/default/ckpt/checkpoint_epoch_6.pth
# Multiple GPUs
bash scripts/dist_test.sh ${NUM_GPUS} --cfgs/waymo_models/mppnet_4frames.yaml  --batch_size  1 \
--ckpt  ../output/waymo_models/mppnet_4frames/default/ckpt/checkpoint_epoch_6.pth
```
To avoid OOM, set BATCH_SIZE=1.

* Test with a memory bank to improve efficiency:
```shell
# Currently, only support 1 GPU with batch_size 1
python test.py --cfg_file cfgs/waymo_models/mppnet_e2e_memorybank_inference.yaml --batch_size 1 \
--ckpt ../output/waymo_models/mppnet_4frames/default/ckpt/checkpoint_epoch_6.pth \
--pretrained_model  ../output/waymo_models/centerpoint_4frames/default/ckpt/checkpoint_epoch_36.pth
```
The default parameters in mppnet_e2e_memorybank_inference.yaml is for 4-frame and just change them to the setting in mppnet_16frames.yaml when using 16-frame.

## Performance
|    Model          | Vec_L1 | Vec_L2 | Ped_L1 | Ped_L2 | Cyc_L1 | Cyc_L2 |  
|:---------------------------------------------:|:----------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|  [centerpoint_4frames](../../tools/cfgs/waymo_models/centerpoint_4frames.yaml) | 76.71/76.17 | 69.13/68.63 | 78.88/75.55 | 71.73/68.61 | 73.73/72.96 | 71.63/70.89 |
|  [mppnet_4frames](../../tools/cfgs/waymo_models/mppnet_4frames.yaml) | 81.54/81.06 | 74.07/73.61 | 84.56/81.94 | 77.20/74.67 | 77.15/76.50 | 75.01/74.38 |
| [mppnet_16frames](../../tools/cfgs/waymo_models/mppnet_16frames.yaml) | 82.74/82.28 | 75.41/74.96 | 84.69/82.25 | 77.43/75.06 | 77.28/76.66 | 75.13/74.52 |

The reported performance of MPPNet is trained with 6 epoch with  `USE_ROI_AUG`  and  `USE_TRAJ_AUG`  flags enabled.