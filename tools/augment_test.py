import mayavi.mlab as mlab

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from visual_utils import visualize_utils as V


def main():
    cfg_from_yaml_file('cfgs/dataset_configs/kitti_dataset.yaml', cfg)

    logger = common_utils.create_logger()

    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg,
        class_names=['Car', 'Pedestrian', 'Cyclist'],
        batch_size=1,
        dist=False,
        workers=4,
        logger=logger,
        training=False,
        merge_all_iters_to_one_epoch=False,
        total_epochs=0
    )

    logger.info(f'Total number of samples: \t{len(train_set)}')

    data_dict_list = []
    for idx, data_dict in enumerate(train_set):
        logger.info(f'Loaded sample index: \t{idx + 1}')
        data_dict = train_set.collate_batch([data_dict])
        data_dict_list.append(data_dict)

    data_dict = data_dict_list[0]
    V.draw_scenes(
        points=data_dict['points'][:, 1:], gt_boxes=data_dict['gt_boxes'][0]
    )
    mlab.show(stop=True)


if __name__ == '__main__':
    main()
