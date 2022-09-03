## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for the installation of `OpenPCDet`.


##Data Preparation
Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) to process the Waymo Open Dataset.

##Training

1.  Train the RPN model  for MPPNet (centerpoint_4frames is employed in the paper)
```shell
bash scripts/dist_train.sh ${NUM_GPUS} --cfg_file cfgs/waymo_models/centerpoint_4frames.yaml
```
The ckpt will be saved in ../output/waymo_models/centerpoint_4frames/default/ckpt.

2.  Save the RPN model's prediction results of  training and val dataset
```shell
# training
bash scripts/dist_test.sh ${NUM_GPUS}  --cfg_file cfgs/waymo_models/mppnet_4frames.yaml \
--ckpt   ../output/waymo_models/centerpoint_4frames/default/ckpt/checkpoint_epoch_36.pth \
--set   DATA_CONFIG.DATA_SPLIT.test train
# val
bash scripts/dist_test.sh ${NUM_GPUS}  --cfg_file cfgs/waymo_models/mppnet_4frames.yaml \
--ckpt   ../output/waymo_models/centerpoint_4frames/default/ckpt/checkpoint_epoch_36.pth \
--set   DATA_CONFIG.DATA_SPLIT.test val
```
The prediction results of train and val dataset will be saved in
../output/waymo_models/centerpoint_4frames/default/eval/epoch_36/train/default/result.pkl,
../output/waymo_models/centerpoint_4frames/default/eval/epoch_36/val/default/result.pkl.

3.  Train MPPNet (using mppnet_4frames as an example)
```shell
bash scripts/dist_train.sh ${NUM_GPUS} --cfg_file cfgs/waymo_models/mppnet_4frames.yaml - --batch_size  2  \
--set DATA_CONFIG.ROI_BOXES_PATH.train  ../output/waymo_models/centerpoint_4frames/default/eval/epoch_36/train/default/result.pkl \
            DATA_CONFIG.ROI_BOXES_PATH.test  ../output/waymo_models/centerpoint_4frames/default/eval/epoch_36/val/default/result.pkl
```
When using 16-frame, we can just change the `cfg_file` to mpppnet_16frames.yaml and the ` DATA_CONFIG.ROI_BOXES_PATH`  is same with 4-frame.
We can also save the paths of train and val results to ROI_BOXES_PATH in mppnet_4frames.yaml and mppnet_16frames.yaml to avoid using the `set` flag.
For each GPU, BATCH_SIZE should be at least equal to 2.  When using 16-frame, the reference GPU memory consumption is 29G with BATCH_SIZE=2.

##Evaluation
* Test with a pretrained model:
```shell
# Single GPU
python test.py --cfg_file cfgs/waymo_models/mppnet_4frames.yaml  --batch_size  1 \
--ckpt  ../output/waymo_models/mppnet_4frames/default/ckpt/checkpoint_epoch_6.pth
# Multiple GPUs
bash scripts/dist_test.sh ${NUM_GPUS} --cfgs/waymo_models/mppnet_4frames.yaml  --batch_size  1 \
--ckpt  ../output/waymo_models/mppnet_4frames/default/ckpt/checkpoint_epoch_6.pth
```
To avoid OOM,  set BATCH_SIZE=1.

* Test with a memory bank to improve efficiency:
```shell
# Currently, only support 1 GPU with batch_size 1
python test.py --cfg_file cfgs/waymo_models/mppnet_e2e_memorybank_inference.yaml --batch_size 1 \
--ckpt ../output/waymo_models/mppnet_4frames/default/ckpt/checkpoint_epoch_6.pth \
--pretrained_model  ../output/waymo_models/centerpoint_4frames/default/ckpt/checkpoint_epoch_36.pth
```
The default parameters in mppnet_e2e_memorybank_inference.yaml is for 4-frame and just change them to the setting in mppnet_16frames.yaml when using 16-frame.
