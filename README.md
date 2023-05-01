# [Windows 11 Compatible]  OpenPCDet 0.6.0

This is a compatible version of OpenPCDet for WIndows 11. 

## OpenPCDet

Official README.md see here: [README_aux.md](README_aux.md)

## Problems known

- Sharedarray removed, Waymo dataset with shared memory may not work.

## Installation schme

- Based on Global CUDA dev 11.4 installed
- IoU3d_nms modified
- Dependency Sharedarray removed

## Environment

- Windows 11
- CUDA-dev 11.4
- cudatoolkit 11.3
- Anaconda2
- Python 3.8.16
- Pytorch 1.10.1
- spconv-cu113 2.3.6
- NVIDIA GeForce RTX 3060 Laptop GPU

## Changelog

[2023-05-1] Dataset shared memory modified.

[2023-04-28] IoU3d_nms modified.

[2023-04-28] Readme and dev branch initialized.

## Modification from official code

- IoU3d_nms
- Part of Dataset Shared Memory

## How to use it

a. Install and configure CUDA 11.4 properly from https://anaconda.org/conda-forge/cudatoolkit-dev/files

b. Clone this repository.

```shell
git clone https://github.com/Uzukidd/OpenPCDet-Win11-Compatible.git
```

c. Create the environment as follows:

```shell
conda env create -f environment.yml
```

c. Install this `pcdet` library and its dependent libraries by running the following command:

```
python setup.py develop
```

## ez check with PointPillar

a. Download `easy_check` from https://drive.google.com/file/d/13czqs5oSTb86QC0ZOMe6wEjgL91_k-VR/view?usp=sharing and extract it into `./tools/`

b. Activate the environment

```shell
conda activate env_openpcdet_base
```

c. Enter tools

```shell
cd ./tools
```

d. Run the demo

```shell
python demo.py --cfg_file ./cfgs/kitti_models/pointpillar.yaml --ckpt ./easy_check/pointpillar_7728.pth --data_path  ./easy_check/000000.bin
```

