import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from ..config import cfg
from .dataset import DatasetTemplate
from .kitti.kitti_dataset import BaseKittiDataset, KittiDataset

__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'BaseKittiDataset': BaseKittiDataset,
    'KittiDataset': KittiDataset
}


def build_dataloader(data_dir, batch_size, dist, workers=4, logger=None, training=True):
    data_dir = Path(data_dir) if os.path.isabs(data_dir) else cfg.ROOT_DIR / data_dir

    dataset = __all__[cfg.DATA_CONFIG.DATASET](
        root_path=data_dir,
        class_names=cfg.CLASS_NAMES,
        split=cfg.MODEL['TRAIN' if training else 'TEST'].SPLIT,
        training=training,
        logger=logger,
    )

    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if dist else None
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0
    )
    return dataset, dataloader, sampler
