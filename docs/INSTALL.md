## Installation

### Requirements
All the codes are tested in the following environment:
* Linux (tested on Ubuntu 14.04/16.04)
* Python 3.6+
* PyTorch 1.1 or higher (tested on PyTorch 1.1)
* CUDA 9.0 or higher
* `spconv v1.0` ([commit 8da6f96](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634))


### Install `pcdet v0.2`
NOTE: Please re-install `pcdet v0.2` by running `python setup.py develop` if you have already installed `pcdet v0.1` previously.

a. Clone this repository.
```shell
git clone https://github.com/open-mmlab/OpenLidarPercept.git
```

b. Install the dependent libraries as follows:

* Install the dependent python libraries: 
```
pip install -r requirements.txt 
```

* Install the SparseConv library, we use the non-official implementation from [`spconv`](https://github.com/traveller59/spconv). 
Note that we use the initial version of `spconv`, make sure you install the `spconv v1.0` ([commit 8da6f96](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634)) instead of the latest one.

c. Install this `pcdet` library by running the following command:
```shell
python setup.py develop
```

## Dataset Preparation

Currently we provide the dataloader of KITTI dataset, and the supporting of more datasets are on the way.  

### KITTI Dataset
* Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows (the road planes could be downloaded from [[road plane]](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing), which are optional for data augmentation in the training):
* NOTE: You need to create the infos and gt database again even you already have them from `pcdet v0.1` 

```
PCDet
├── data
│   ├── kitti
│   │   │──ImageSets
│   │   │──training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │──testing
│   │   │   ├──calib & velodyne & image_2
├── pcdet
├── tools
```

* Generate the data infos by running the following command: 
```python 
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```
