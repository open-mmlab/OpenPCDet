# OpenPCDet on Modal

This is a guide to running PCDet on [Modal](https://modal.com/), a platform for running machine learning workflows on the cloud.

In this example we will train a pointpillar model on the nuScenes mini dataset. Credit to [ies0411](https://github.com/ies0411) who contributed the dockerfile in https://github.com/open-mmlab/OpenPCDet/pull/1513.

## Install modal

* `pip install modal`
* `modal setup`

See https://modal.com/docs/guide for more information.

## Prepare data

* Download the nuScenes mini dataset
* Create shared Modal volume: `modal volume create nuscenes`.
* Optionally: delete all image and radar folders to speed up the next step.
* Upload data to the volume: `cd path-to-downloaded-data`, `modal volume put nuscenes v1.0-mini`.

## Format the data

Build the nuScenes `infos` used by OpenPCDet. This is a one-time operation.

* `modal run setup_nuscenes.py`

## Train the model

* `modal run train_pp.py`
