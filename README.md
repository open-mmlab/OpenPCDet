# Object Relations for 3D Object Detection

This work focuses on exploring the impact of modeling object relation in two-stage object detection pipelines, aiming to enhance their detection performance. It extends OpenPCDet with a module that models object relations which can be integrated into existing object detectors. To get this project running please check [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).
 

## Project Structure

This project extends [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). Some of OpenPCDet model's are extended with an Object Relation Module. See the list below.

## Installation

1. Build the provided Dockerfile
2. Run the following command in the project root
```bash
python setup.py develop
```
3. Prepare data according to [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)

## Training & Testing
The following commands should be ran in ``./tools``.

Train Model:
```bash
python train.py --cfg_file {PATH_TO_CONFIG_FILE}
```

Test Model:
```bash
python test.py --cfg_file {PATH_TO_CONFIG_FILE} --ckpt {PATH_TO_MODEL}
```

## Models

| Model                                 | Path                                                                      | Description                                                               | Dataset |
| :------------------------------------ | :------------------------------------------------------------------------ | :------------------------------------------------------------------------ | :------ |
| **PV-RCNN-Relation**                  | `tools/cfgs/kitti_models/pv_rcnn_relation{_car_class_only}.yaml`          | PV-RCNN model extended with the Object Relation Module.                   | KITTI   |
| **PV-RCNN-Relation**                  | `tools/cfgs/waymo_models/pv_rcnn_relation.yaml`          | PV-RCNN model extended with the Object Relation Module.                   | Waymo   |
| **PV-RCNN++-Relation**                | `tools/cfgs/kitti_models/pv_rcnn_plusplus_relation{_car_class_only}.yaml` | PV-RCNN++ model extended with the Object Relation Module.                 | KITTI   |
| **Voxel-RCNN-Relation Car Class**     | `tools/cfgs/kitti_models/voxel_rcnn_relation_car_class_only.yaml`         | Voxel-RCNN extended with the Object Relation Module.                      | KITTI   |
| **PartA2-Relation Car Class**         | `tools/cfgs/kitti_models/PartA2_relation_car_class_only.yaml`             | PartA2 model extended with the Object Relation Module, trained only on the car class. | KITTI   |


For some models the suffix '_car_class_only.yaml' can be used to train the model only on the car class






## Motivation

There are four motivations for modeling object relations in the refinement stage of two-stage object detection pipelines.

- **Detecting Occlusions:** If information between an occluded and an occluder object is shared, the occluded object can be informed about its occlusion status. This can help the model learn the difference between sparse proposals that are heavily occluded and noise in the data.

- **Exploiting Patterns:** Traffic scenes often follow specific patterns that can be exploited by the object detector.

- **Increase of Receptive Fields:** Current object detectors fail to incorporate enough context in the refinement stage because their receptive fields are too small. Object relations can be seen as an efficient mechanism to increase the receptive field.

- **Proposal Consensus:** Proposals often form clusters around potential objects. Each proposal might have a different view of the object. Sharing information between these proposals leads to a consensus prediction.


| ![Image 1](resources/occlusion.png) | ![Image 2](resources/pattern.png) |
|:-:|:-:|
| *Detecting Occlusions*      | *Exploiting Patterns*      |
| ![Image 3](resources/radius.png) | ![Image 4](resources/proposal_consensus.png)
| *Increase of Receptive Fields*      | *Proposals Consensus*      |


# PV-RCNN-Relation

PV-RCNN-Relation is an implementation of an object relations module applied to the PVRCNN baseline. It beats its baseline on all difficulties for the vehicle class.


## Results

| Model             | Easy R11 / R40 | Moderate R11 / R40 | Hard R11 / R40 | All R11 / R40 |
|-------------------|----------------|--------------------|----------------|---------------|
| PV-RCNN            | 89.39 / 92.02  | 83.63 / 84.80      | 78.86 / 82.58  | 83.91 / 86.45 |
| PV-RCNN-Relation   | 89.59 / 92.53  | 84.56 / 85.22      | 79.04 / 82.99  | 84.35 / 86.90 |
| *Improvement*     | **+0.20 / +0.51** | **+0.93 / +0.42** | **+0.18 / +0.41** | **+0.44 / +0.45** |

*Comparison of PVRCNN and PVRCNN-Relation on KITTI validation set. Trained and evaluated only on the car class. All models were trained for 80 epochs and the best-performing epoch per model and metric was chosen. **Both models were trained three times** and the average is reported. The* Improvement *row represents the difference in mAP between the two models.*


| | |
|:-------------------------:|:-------------------------:|
| ![Image 1](resources/side.png) | ![Image 2](resources/relation_side.png) |
|  |  |

Qualitative results for PV-RCNN baseline and PV-RCNN-Relation on Waymo data. Predictions are shown in green, labels in blue, and edges that connect proposals to share information in red. 







