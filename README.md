# OpenLidarPercept 

## Introduction
`OpenLidarPercept` is an open source project for LiDAR-based 3D scene perception. 
As of now, it mainly consists of `PCDet` toolbox for 3D object detection from point cloud.  


### What does `PCDet` toolbox do?

`PCDet` is a general PyTorch-based codebase for 3D object detection from point cloud. 
It currently supports multiple state-of-the-art 3D object detection methods with highly refactored codes for both one-stage and two-stage 3D detection frameworks.

It is also the official code release of [`[Part-A^2 net]`](https://arxiv.org/abs/1907.03670) and [`[PV-RCNN]`](https://arxiv.org/abs/1912.13192). 

We are actively updating this repo currently, and more datasets and models will be supported soon. 
Contributions are also welcomed. 


### Currently Supported Features
- [x] Unified point cloud coordinate and clear code structure and for supporting lots of datasets and approaches
- [x] Support both one-stage and two-stage 3D object detection frameworks
- [x] Support distributed training & testing with multiple GPUs and multiple machines
- [x] Support multiple heads on different scales to detect different classes
- [x] Support stacked version set abstraction to encode various number of points in different scenes
- [x] Support Adaptive Training Sample Selection (ATSS) for target assignment
- [x] Support RoI-aware point cloud pooling & RoI-grid point cloud pooling
- [x] Support GPU version 3D IoU calculation and rotated NMS 

## Model Zoo

### KITTI 3D Object Detection Baselines
Selected supported methods are shown in the below table. The results are the 3D detection performance of car class on the *val* set of KITTI dataset.
All models are trained with 8 GPUs and are available for download.

|                                             | Batch Size | AP_Easy | AP_Mod. | AP_Hard | download  |
|---------------------------------------------|:----------:|:-------:|:-------:|:-------:|:---------:|
| [PointPillar](tools/cfgs/kitti_models/pointpillar.yaml) | 32 | - | - | - | [model]() | 
| [SECOND](tools/cfgs/kitti_models/second.yaml)           | 32  | - | - | - | [model]() |
| [Part-A^2](tools/cfgs/kitti_models/PartA2.yaml)    | 32 | - | - | - | [model]() |
| [PV-RCNN](tools/cfgs/kitti_models/pv_rcnn.yaml) | 16 | - | - | - | [model]() |
| [SECOND-MultiHead](tools/cfgs/kitti_models/second_multihead.yaml) | 32 | - | - | - | - |

## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for installation and dataset preparation.


## Get Started

Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) to learn more usage about this project.

## License

`OpenLidarPercept` is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement
`OpenLidarPercept` is an open source project for LiDAR-based 3D scene perception that supports multiple
LiDAR-based perception models as shown above. Some parts of `PCDet` are learned from the official released codes of the above supported methods. 
We would like to thank for their proposed methods and the official implementation.   

We hope that this repo could serve as a strong and flexible codebase to benefit the research community by speeding up the process of reimplementing previous works and/or developing new methods.


## Citation 
If you find this project useful in your research, please consider cite:


```
@inproceedings{shi2020pv,
  title={Pv-rcnn: Point-voxel feature set abstraction for 3d object detection},
  author={Shi, Shaoshuai and Guo, Chaoxu and Jiang, Li and Wang, Zhe and Shi, Jianping and Wang, Xiaogang and Li, Hongsheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10529--10538},
  year={2020}
}


@article{shi2020points,
  title={From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network},
  author={Shi, Shaoshuai and Wang, Zhe and Shi, Jianping and Wang, Xiaogang and Li, Hongsheng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2020},
  publisher={IEEE}
}

@inproceedings{shi2019pointrcnn,
  title={PointRCNN: 3d Object Progposal Generation and Detection from Point Cloud},
  author={Shi, Shaoshuai and Wang, Xiaogang and Li, Hongsheng},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={770--779},
  year={2019}
}
```

## Contact
This project is currently maintained by Shaoshuai Shi ([@sshaoshuai](http://github.com/sshaoshuai)) and Chaoxu Guo ([@Gus-Guo](https://github.com/Gus-Guo)).
