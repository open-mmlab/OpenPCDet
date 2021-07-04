import torch
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler

from pcdet.utils import common_utils

from .dataset import DatasetTemplate
from .once.once_dataset import ONCEDataset

from .once.once_semi_dataset import ONCEPretrainDataset, ONCELabeledDataset, ONCEUnlabeledDataset, ONCETestDataset, split_once_semi_data

__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'ONCEDataset': ONCEDataset
}

_semi_dataset_dict = {
    'ONCEDataset': {
        'PARTITION_FUNC': split_once_semi_data,
        'PRETRAIN': ONCEPretrainDataset,
        'LABELED': ONCELabeledDataset,
        'UNLABELED': ONCEUnlabeledDataset,
        'TEST': ONCETestDataset
    }
}

class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def build_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0):

    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=training,
        logger=logger,
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0
    )

    return dataset, dataloader, sampler

def build_semi_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4,
                     logger=None, merge_all_iters_to_one_epoch=False):

    assert merge_all_iters_to_one_epoch is False

    train_infos, test_infos, labeled_infos, unlabeled_infos = _semi_dataset_dict[dataset_cfg.DATASET]['PARTITION_FUNC'](
        info_paths = dataset_cfg.INFO_PATH,
        data_splits = dataset_cfg.DATA_SPLIT,
        root_path = root_path,
        labeled_ratio = dataset_cfg.LABELED_RATIO,
        logger = logger
    )

    pretrain_dataset = _semi_dataset_dict[dataset_cfg.DATASET]['PRETRAIN'](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        infos = train_infos,
        root_path=root_path,
        logger=logger,
    )
    if dist:
        pretrain_sampler = torch.utils.data.distributed.DistributedSampler(pretrain_dataset)
    else:
        pretrain_sampler = None

    pretrain_dataloader = DataLoader(
        pretrain_dataset, batch_size=batch_size['pretrain'], pin_memory=True, num_workers=workers,
        shuffle=(pretrain_sampler is None) and True, collate_fn=pretrain_dataset.collate_batch,
        drop_last=False, sampler=pretrain_sampler, timeout=0
    )

    labeled_dataset = _semi_dataset_dict[dataset_cfg.DATASET]['LABELED'](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        infos = labeled_infos,
        root_path=root_path,
        logger=logger,
    )
    if dist:
        labeled_sampler = torch.utils.data.distributed.DistributedSampler(labeled_dataset)
    else:
        labeled_sampler = None
    labeled_dataloader = DataLoader(
        labeled_dataset, batch_size=batch_size['labeled'], pin_memory=True, num_workers=workers,
        shuffle=(labeled_sampler is None) and True, collate_fn=labeled_dataset.collate_batch,
        drop_last=False, sampler=labeled_sampler, timeout=0
    )

    unlabeled_dataset = _semi_dataset_dict[dataset_cfg.DATASET]['UNLABELED'](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        infos = unlabeled_infos,
        root_path=root_path,
        logger=logger,
    )
    if dist:
        unlabeled_sampler = torch.utils.data.distributed.DistributedSampler(unlabeled_dataset)
    else:
        unlabeled_sampler = None
    unlabeled_dataloader = DataLoader(
        unlabeled_dataset, batch_size=batch_size['unlabeled'], pin_memory=True, num_workers=workers,
        shuffle=(unlabeled_sampler is None) and True, collate_fn=unlabeled_dataset.collate_batch,
        drop_last=False, sampler=unlabeled_sampler, timeout=0
    )

    test_dataset = _semi_dataset_dict[dataset_cfg.DATASET]['TEST'](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        infos = test_infos,
        root_path=root_path,
        logger=logger,
    )
    if dist:
        rank, world_size = common_utils.get_dist_info()
        test_sampler = DistributedSampler(test_dataset, world_size, rank, shuffle=False)
    else:
        test_sampler = None
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size['test'], pin_memory=True, num_workers=workers,
        shuffle=(test_sampler is None) and False, collate_fn=test_dataset.collate_batch,
        drop_last=False, sampler=test_sampler, timeout=0
    )

    datasets = {
        'pretrain': pretrain_dataset,
        'labeled': labeled_dataset,
        'unlabeled': unlabeled_dataset,
        'test': test_dataset
    }
    dataloaders = {
        'pretrain': pretrain_dataloader,
        'labeled': labeled_dataloader,
        'unlabeled': unlabeled_dataloader,
        'test': test_dataloader
    }
    samplers = {
        'pretrain': pretrain_sampler,
        'labeled': labeled_sampler,
        'unlabeled': unlabeled_sampler,
        'test': test_sampler
    }

    return datasets, dataloaders, samplers