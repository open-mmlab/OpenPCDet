
## Installation

Please refer to [INSTALL.md](../INSTALL.md) for the installation of `OpenPCDet`.
* We recommend the users to check the version of pillow and use pillow==8.4.0 to avoid bug in bev pooling.

## Data Preparation
Please refer to [GETTING_STARTED.md](../GETTING_STARTED.md) to process the multi-modal Nuscenes Dataset.

## Training

1.  Train the lidar branch for BEVFusion:
```shell
bash scripts/dist_train.sh ${NUM_GPUS} --cfg_file cfgs/nuscenes_models/transfusion_lidar.yaml \
```
The ckpt will be saved in ../output/nuscenes_models/transfusion_lidar/default/ckpt, or you can download pretrained checkpoint directly form [here](https://drive.google.com/file/d/1cuZ2qdDnxSwTCsiXWwbqCGF-uoazTXbz/view?usp=share_link).

2.  To train BEVFusion, you need to download pretrained parameters for image backbone [here](https://drive.google.com/file/d/1v74WCt4_5ubjO7PciA5T0xhQc9bz_jZu/view?usp=share_link), and specify the path in [config](../../tools/cfgs/nuscenes_models/bevfusion.yaml#L88). Then run the following command:
```shell
bash scripts/dist_train.sh ${NUM_GPUS} --cfg_file cfgs/nuscenes_models/bevfusion.yaml \
--pretrained_model path_to_pretrained_lidar_branch_ckpt \
```
## Evaluation
* Test with a pretrained model:
```shell
bash scripts/dist_test.sh ${NUM_GPUS} --cfg_file  cfgs/nuscenes_models/bevfusion.yaml \
--ckpt ../output/cfgs/nuscenes_models/bevfusion/default/ckpt/checkpoint_epoch_6.pth
```

## Performance
All models are trained with spconv 1.0, but you can directly load them for testing regardless of the spconv version.
|                                                                                                    |   mATE |  mASE  |  mAOE  | mAVE  | mAAE  |  mAP  |  NDS   |                                              download                                              | 
|----------------------------------------------------------------------------------------------------|-------:|:------:|:------:|:-----:|:-----:|:-----:|:------:|:--------------------------------------------------------------------------------------------------:|
| [TransFusion-L](../../tools/cfgs/nuscenes_models/transfusion_lidar.yaml)   |  27.96 | 	25.37 | 	29.35 | 27.31 | 18.55 | 64.58 | 69.43  | [model-32M](https://drive.google.com/file/d/1cuZ2qdDnxSwTCsiXWwbqCGF-uoazTXbz/view?usp=share_link) |
| [BEVFusion](../../tools/cfgs/nuscenes_models/bevfusion.yaml)   |  28.03 | 	25.43 | 	30.19 | 26.76 | 18.48 | 67.75 | 70.98  | [model-157M](https://drive.google.com/file/d/1X50b-8immqlqD8VPAUkSKI0Ls-4k37g9/view?usp=share_link) |
