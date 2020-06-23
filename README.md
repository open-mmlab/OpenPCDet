# OpenLidarPercept 

## Introduction
`OpenLidarPercept` is an open source project for LiDAR-based 3D scene perception. 
As of now, it mainly consists of the `PCDet` toolbox for 3D object detection from point cloud.  


### What does `PCDet` toolbox do?

`PCDet` is a general PyTorch-based codebase for 3D object detection from point cloud. 
It currently supports multiple state-of-the-art 3D object detection methods (`PointPillar`, `SECOND`, `Part-A^2 Net`, `PV-RCNN`) with highly refactored codes for both one-stage and two-stage frameworks.

It is also the official code release of [`[Part-A^2 net]`](https://arxiv.org/abs/1907.03670) and [`[PV-RCNN]`](https://arxiv.org/abs/1912.13192). 


### Currently Supported Features
- [x] Support both one-stage and two-stage 3D object detection frameworks
- [x] Distributed training & testing with multiple GPUs and multiple machines
- [x] Clear code structure and unified point cloud coordinate for supporting more datasets and approaches
- [x] RoI-aware point cloud pooling & RoI-grid point cloud pooling
- [x] Stacked version set abstraction to support various number of points in differnet scenes
- [x] GPU version 3D IoU calculation and rotated NMS 

