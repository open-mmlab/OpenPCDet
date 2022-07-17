For Custom Dataset using
## Custom Dataset
For pure point cloud dataset, which means you don't have images generated when got point cloud data from a self-defined scene. Label those raw data and make sure label files to be kitti-like:
```
Car 0 0 0 0 0 0 0 1.50 1.46 3.70 -5.12 1.85 4.13 1.56
Pedestrian 0 0 0 0 0 0 0 1.54 0.57 0.41 -7.92 1.94 15.95 1.57
DontCare 0 0 0 0 0 0 0 -1 -1 -1 -1000 -1000 -1000 -10
```
Some items (which is shown from the first zero to the seventh zero above) are not necessary because they are meaningless if no cameras. And the `image` folder, `calib` folder are both needless, which will be much more convenient for not using just official dataset. The point cloud dataset should be `.bin` format.

Place the custom dataset:
```
OpenPCDet
├── data
│   ├── custom
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──velodyne & label_2
│   │   │── testing
│   │   │   ├──velodyne
├── pcdet
├── tools
```
## Calibration
Calibration rules for cameras are not need. But you need to define how to transform from KITTI coordinates to lidar coordinates. The lidar coordinates are the custom coordinates. The raw data are in lidar coordinates and the labels are in KITTI coordinates. This self-defined transform is written in `custom_dataset->get_calib (188)` which is used to get gt_boxes from labels.
## Other configurations
Possible other parameters or names that need to be check to adapt the custom scene.
- config files
 ```
 CLASS_NAMES: ['Car', 'Pedestrian', 'Cyclist']  # pv_rcnn.yaml
 ...
'anchor_sizes': [[3.9, 1.6, 1.56]], # pv_rcnn.yaml
...
POINT_CLOUD_RANGE: [-70.4, -40, -3, 70.4, 40, 1] # custom_dataset.yaml
...
 ```
The train, test and pred are all the same as others.