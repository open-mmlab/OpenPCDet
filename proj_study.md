
#### pcdet

pcdet主体框架

- datasets // Data processing/loader and dataset-specific evaluation tools
  - __init__.py
  - dataset.py
  - augmentor
    - augmentor_utils.py
    - data_augmentor.py
    - database_sampler.py
  - processor
    - data_processor.py
    - point_feature_encoder.py
  - kitti
    - kitti_object_eval_python // kitti的python评估脚本
      - eval.py
      - evaluate.py
      - kitti_common.py
      - rotate_iou.py
      - LICENSE
      - README.md
    - kitti_dataset.py
    - kitti_utils.py
  - nuscenes
    - nuscenes_dataset.py
    - nuscenes_utils.py
  - waymo
    - waymo_dataset.py
    - waymo_eval.py
    - waymo_utils.py
  

- models // Model definition and forward/backward codes
  - __init__.py
  - backbones_2d
    - __init__.py
    - base_bev_backbone.py
    - map_to_bev
      - __init__.py
      - height_compression.py
      - pointpillar_scatter.py
  - backbones_3d
    - __init__.py
    - pointnet2_backbone.py
    - spconv_backbone.py
    - spconv_unet.py
    - pfe
      - __init__.py
      - voxel_set_abstraction.py
    - vfe
      - __init__.py
      - mean_vfe.py
      - pillar_vef.py
      - vfe_template.py
  - dense_heads
    - __init__.py
    - anchor_head_multi.py
    - anchor_head_single.py
    - anchor_head_template.py
    - point_head_box.py
    - point_head_simple.py
    - point_head_template.py
    - point_intra_part_head.py
    - target_assigner
      - anchor_generator.py
      - atss_target_assigner.py
      - axis_aligned_target_assigner.py
  - detectors
    - __init__.py
    - detector3d_template.py
    - PartA2_net.py
    - point_rcnn.py
    - pointpillar.py
    - pv_rcnn.py
    - second_net.py
  - model_utils
    - model_nms_utils.py
  - roi_heads
    - __init__.py
    - partA2_head.py
    - pointrcnn_head.py
    - pvrcnn_head.py
    - roi_head_template.py
    - target_assigner
      - proposal_target_layer.py

- ops // Customized operations (C++/CUDA codes) for 3D detection
  - iou3d_nms
  - pointnet2
  - roiware_pool3d
  - roipoint_pool3d

- utils // Common utilized codes and loss utilized codes for 3D detection
  - box_coder_utils.py
  - box_utils.py
  - calibration_kitti.py
  - common_utils.py
  - loss_utils.py
  - object3d_kitti.py


- __init__.py

- config.py 配置解析工具
  - log_config_to_file(cfg, pre='cfg', logger=None)
  - cfg_from_list(cfg_list, config)
  - merge_new_config(config, new_config)
  - cfg_from_yaml_file(cfg_file, config)


#### tools

训练、评估、可视化工具以及相关配置文件

- cfgs 各种模型独立的配置文件以及各数据集独立的配置文件
  - dataset_configs
    - kitti_dataset.yaml
    - nuscenes_dataset.yaml
    - waymo_dataset.yaml
  - kitti_models
    - PartA2_free.yaml
    - PartA2.yaml
    - pointpillar.yaml
    - pointrcnn_iou.yaml
    - pointrcnn.yaml
    - second_multihead.yaml
    - second.yaml
  - nuscenes_models
    - cbgs_pp_multihead.yaml
    - cbgs_second_multihead.yaml
  - waymo_models
    - PartA2.yaml
    - pv_rcnn.yaml
    - second.yaml
  
- scripts 训练与测试使用到的脚本文件
  - dist_test.sh
  - dist_train.sh
  - slurm_test_mgpu.sh
  - slurm_test_single.sh
  - slurm_train.sh

- eval_utils 评估工具
  - statistics_info(cfg, ret_dict, metric, disp_dict)
  - eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None)
  
- train_utils 训练工具
  - optimization
    - __init__.py
    - fastai_optim.py
    - learning_schedules_fastai.py
  - train_utils.py
    - train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg, rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False)
    - train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg, start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None, lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50, merge_all_iters_to_one_epoch=False)
    - model_state_to_cpu(model_state)
    - checkpoint_state(model=None, optimizer=None, epoch=None, it=None)
    - save_checkpoint(state, filename='checkpoint')

- visual_utils 可视化工具
  - check_numpy_to_torch(x)
  - rotate_points_along_z(points, angle)
  - boxes_to_corners_3d(boxes3d)
  - visualize_pts(pts, fig=None, bgcolor=(0, 0, 0), fgcolor=(1.0, 1.0, 1.0), show_intensity=False, size=(600, 600), draw_origin=True)
  - draw_sphere_pts(pts, color=(0, 1, 0), fig=None, bgcolor=(0, 0, 0), scale_factor=0.2)
  - draw_grid(x1, y1, x2, y2, fig, tube_radius=None, color=(0.5, 0.5, 0.5))
  - draw_multi_grid_range(fig, grid_size=20, bv_range=(-60, -60, 60, 60))
  - draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_scores=None, ref_labels=None)
  - draw_corners3d(corners3d, fig, color=(1, 1, 1), line_width=2, cls=None, tag='', max_num=500, tube_radius=None)

- train.py 训练入口
  - parse_config()
  - main()

- test.py 测试入口
  - parse_config()
  - eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False)
  - get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args)
  - repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=False)
  - main()

- demo.py 样例
  - class DemoDataset(DatasetTemplate)
  - parse_config()
  - main()



#### docs

存放安装、使用等教程文档与项目模块与框架图


#### data

存放数据集以及train/val splits
- kitti
- waymo


#### setup.py

用于编译安装C++/cuda扩展，包括：
- iou3d_nms
- roiware_pool3d
- roipoint_pool3d
- pointnet2_stack
- pointnet2_batch


#### others

- .gitignore
- LICENSE
- README.md
- requirements.txt