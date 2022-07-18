## Custom Dataset
For pure point cloud dataset, which means you don't have images generated when got point cloud data from a self-defined scene, use `custom_dataset.py` to load it. Label those raw data and make sure label files to be kitti-format:
```
Car 0 0 0 0 0 0 0 1.50 1.46 3.70 -5.12 1.85 4.13 1.56
Pedestrian 0 0 0 0 0 0 0 1.54 0.57 0.41 -7.92 1.94 15.95 1.57
DontCare 0 0 0 0 0 0 0 -1 -1 -1 -1000 -1000 -1000 -10
```
Some items (which is shown from the first zero to the seventh zero above) are not necessary because they are meaningless if no cameras. And the `image` folder, `calib` folder are both needless, which will be much more convenient for not using just official dataset. The point cloud dataset should be `.bin` format, other formats can be supported by modifying relative codes.

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
Calibration rules for cameras are not need. But you need to define how to transform your custom data from KITTI coordinates to lidar coordinates. The lidar coordinates are the custom coordinates. The raw data are in lidar coordinates while the labels are in KITTI coordinates (If using some professtional tools like labelCloud, it generate directly kitti-format labels when you finish putting the boxes. If not, the labels and raw data should be in the same coordinates before training so make sure this). This self-defined transform is written in `custom_dataset->get_calib` which is used to get gt_boxes from labels.

For an example:
Use labelCloud to label the custom dataset, and preset the format to `kitti`, the coordinates of labels and the coordinates or custom data (which is seen on labeling window) are possibly different because an implicit conversion will be made when generating the labels. To find this difference, go to see the code which do this conversion.

Relative code in labelCloud tool is https://github.com/ch-sa/labelCloud/blob/badf241b618c42894f6c711b8b1a6adcbfffee11/labelCloud/io/labels/kitti.py#L34-L38 (import labels) and https://github.com/ch-sa/labelCloud/blob/badf241b618c42894f6c711b8b1a6adcbfffee11/labelCloud/io/labels/kitti.py#L58-L63 (export labels). Notice that import and export labels correspond to opposite transformations, to know how to transform kitti coordinates to custom coordinates you should refer to `import labels`:
```python
if self.transformed:
    centroid = centroid[2], -centroid[0], centroid[1] - 2.3
dimensions = [float(v) for v in line_elements[8:11]]
if self.transformed:
    dimensions = dimensions[2], dimensions[1], dimensions[0]
```
The code above shows `import labels` in labelCloud. And relatively code of `get_calib` are shown below:
```python
loc_lidar = np.concatenate([np.array((float(loc_obj[2]), float(-loc_obj[0]), float(loc_obj[1]-2.3)), dtype=np.float32).reshape(1,3) for loc_obj in loc])
```
## Other configurations
Possible other parameters or names that need to be check to adapt the custom scene.
- config files. The class names, anchor sizes or point cloud range etc.
 ```
 CLASS_NAMES: ['Car', 'Pedestrian', 'Cyclist']
 ...
'anchor_sizes': [[3.9, 1.6, 1.56]]
...
POINT_CLOUD_RANGE: [-70.4, -40, -3, 70.4, 40, 1] # custom_dataset.yaml
...
 ```
The train, test and pred are all the same as other datasets.