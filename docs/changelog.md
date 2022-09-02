# Changelog and Guidelines

### [2022-09-02] Update to v0.6.0:

* How to process data to support multi-frame training/testing on Waymo Open Dataset?
   * If you never use the OpenPCDet, you can directly follow the [GETTING_STARTED.md](GETTING_STARTED.md)
   * If you have been using previous OpenPCDet (`v0.5`), then you need to follow the following steps to update your data:
       * Update your waymo infos  (the `*.pkl` files for each sequence) by adding argument `--update_info_only`:
        ```
        python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos --cfg_file tools/cfgs/dataset_configs/waymo_dataset.yaml --update_info_only
        ```   
       * Generate multi-frame GT database for copy-paste augmentation of multi-frame training. There is also a faster version with parallel data generation by adding `--use_parallel`, but you need to read the codes and rename the file after getting the results.
        ```
        python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_gt_database --cfg_file tools/cfgs/dataset_configs/waymo_dataset_multiframe.yaml 
        ```
        This will generate the new files like the following (the last three lines under `data/waymo`): 

```
OpenPCDet
├── data
│   ├── waymo
│   │   │── ImageSets
│   │   │── raw_data
│   │   │   │── segment-xxxxxxxx.tfrecord
|   |   |   |── ...
|   |   |── waymo_processed_data_v0_5_0
│   │   │   │── segment-xxxxxxxx/
|   |   |   |── ...
│   │   │── waymo_processed_data_v0_5_0_gt_database_train_sampled_1/
│   │   │── waymo_processed_data_v0_5_0_waymo_dbinfos_train_sampled_1.pkl
│   │   │── waymo_processed_data_v0_5_0_gt_database_train_sampled_1_global.npy (optional)
│   │   │── waymo_processed_data_v0_5_0_infos_train.pkl (optional)
│   │   │── waymo_processed_data_v0_5_0_infos_val.pkl (optional)
|   |   |── waymo_processed_data_v0_5_0_gt_database_train_sampled_1_multiframe_-4_to_0 (new)
│   │   │── waymo_processed_data_v0_5_0_waymo_dbinfos_train_sampled_1_multiframe_-4_to_0.pkl (new)
│   │   │── waymo_processed_data_v0_5_0_gt_database_train_sampled_1_multiframe_-4_to_0_global.np  (new, optional)
 
├── pcdet
├── tools
```
