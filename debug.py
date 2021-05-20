# %%
import os
import mayavi.mlab as mlab
import numpy as np
from copy import deepcopy
import pickle
import logging


import pcdet.datasets.augmentor

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from tools.visual_utils import visualize_utils as V

mlab.init_notebook()


# %%
# generate pickles from the kitti data
# os.system("python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml")

# %% [markdown]
# ## Visualizing the LiDAR point cloud with labels

# %%
logger = common_utils.create_logger(log_level=logging.DEBUG)


# %%
cfg_from_yaml_file('tools/cfgs/dataset_configs/kitti_dataset.yaml', cfg)
# cfg_from_yaml_file('tools/cfgs/kitti_models/pointpillar_augs.yaml', cfg)

cfg.DATA_PATH = 'data/kitti'

train_set, train_loader, train_sampler = build_dataloader(
    dataset_cfg=cfg,
    class_names=['Car', 'Pedestrian', 'Cyclist'],
    batch_size=1,
    dist=False,
    workers=4,
    logger=logger,
    training=True,
    merge_all_iters_to_one_epoch=False,
    total_epochs=0
)

logger.info(f'Total number of samples: \t{len(train_set)}')

data_dict_list = []
logger.info('Loading samples')
for idx, data_dict in enumerate(train_set):
    logger.info(f'Loaded sample index: \t{idx+1}')
    data_dict = train_set.collate_batch([data_dict])
    data_dict_list.append(data_dict)
    #if idx > 8: break


train_set[2]
# %%
scene = data_dict_list[2]


# %%
scene