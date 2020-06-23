import torch
from torch.utils.data import DataLoader
from .dataset import DatasetTemplate
from .kitti.kitti_dataset import KittiDataset


__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'KittiDataset': KittiDataset,
}


def build_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4,
                     logger=None, training=True):

    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
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
