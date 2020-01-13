import os
from path import Path
import torch
from torch.utils.data import DataLoader
from ..config import cfg
from .dataset import get_dataset_class
from .kitti_dataset import KittiDataset


def build_dataloader(data_dir, batch_size, dist, workers=4, logger=None):
    data_dir = Path(data_dir) if os.path.isabs(data_dir) else cfg.ROOT_DIR / data_dir
    train_set = get_dataset_class(cfg.DATA_CONFIG.DATASET)(
        root_path=data_dir,
        class_names=cfg.CLASS_NAMES,
        split=cfg.MODEL.TRAIN.SPLIT,
        training=True,
        logger=logger,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set) if dist else None
    train_loader = DataLoader(
        train_set, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(train_sampler is None), collate_fn=train_set.collate_batch,
        drop_last=False, sampler=train_sampler, timeout=0
    )
    test_loader = None
    return train_set, train_loader, test_loader, train_sampler
