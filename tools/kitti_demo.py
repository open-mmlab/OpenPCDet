import os
import torch
import json

from pcdet.models import build_network, load_data_to_gpu
from pathlib import Path
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets.kitti.kitti_dataset import KittiDataset

# import mayavi.mlab as mlab
# from visual_utils import visualize_utils as V

NUMBER_OF_SCENES = 500

def main(cfg_path, model_path, save_3d=False, tag=None):
    cfg_from_yaml_file(cfg_path, cfg)
    logger = common_utils.create_logger()
    logger.info('-----------------Creating data for visualization-------------------------')
    kitti_dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, root_path=Path(cfg.DATA_CONFIG.DATA_PATH),
    )
    
    logger.info(f'Total number of samples: \t{NUMBER_OF_SCENES}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=kitti_dataset)
    model.load_params_from_file(filename=model_path, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    # create folder for visualization
    vis_path = '/'.join(os.path.normpath(model_path).split(os.path.sep)[:-2]) + '/visualization' + tag
    os.makedirs(vis_path, exist_ok=True)

    with torch.no_grad():
        for idx, data_dict in enumerate(kitti_dataset):
            if idx >= NUMBER_OF_SCENES:
                break
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = kitti_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            
            if save_3d:
                torch.save(data_dict['points'][:,1:], os.path.join(vis_path, 'points_{}.pt'.format(int(data_dict['frame_id']))))
                torch.save(pred_dicts[0]['pred_boxes'], os.path.join(vis_path, 'pred_boxes_{}.pt'.format(int(data_dict['frame_id']))))
                torch.save(pred_dicts[0]['pred_scores'], os.path.join(vis_path, 'pred_scores_{}.pt'.format(int(data_dict['frame_id']))))
                torch.save(pred_dicts[0]['pred_labels'], os.path.join(vis_path, 'pred_labels_{}.pt'.format(int(data_dict['frame_id']))))
                torch.save(data_dict['gt_boxes'], os.path.join(vis_path, 'gt_boxes_{}.pt'.format(int(data_dict['frame_id']))))
                if 'gnn_edges_final' in pred_dicts[0]:
                    torch.save(pred_dicts[0]['gnn_edges_final'],os.path.join(vis_path, 'gnn_edges{}.pt'.format(int(data_dict['frame_id']))))
                    json.dump(pred_dicts[0]['edge_to_pred'] , open(os.path.join(vis_path, 'edge_to_predict{}.json'.format(int(data_dict['frame_id']))), 'w'))
                    
            else:
                # fig = V.draw_scenes(
                #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
                # )
                # mlab.savefig(os.path.join(vis_path, 'points_{}.pt'.format(int(data_dict['frame_id']))))
                pass

    logger.info('Demo done.')

if __name__ == '__main__':
    # model_path = '../output/cfgs/kitti_models/pv_rcnn_relation/2023-09-15_10-21-38'
    model_path = '../output/cfgs/kitti_models/pv_rcnn_relation_car_class_only/2023-09-29_07-21-48'
    # model_path = '../output/cfgs/kitti_models/pv_rcnn_relation/2023-08-25_13-47-22'
    # full_model_path = model_path + '/ckpt/checkpoint_epoch_73.pth'
    full_model_path = model_path + '/ckpt/checkpoint_epoch_80.pth'
    # cfg_path = model_path + '/pv_rcnn_relation.yaml'
    cfg_path = '../tools/cfgs/kitti_models/pv_rcnn_relation_car_class_only.yaml'
    # /pv_rcnn_relation.yaml
    tag = '/epoch_80/'
    # tag = '/no_post_processing/'
    # tag = '/no_post_processing-94_epoch/'
    # tag = '/epoch_100_no_post_processing/'

    # model_path = '../output/cfgs/kitti_models/pv_rcnn/2023-08-01_20-06-45/ckpt/checkpoint_epoch_90.pth'
    # model_path = '../output/cfgs/kitti_models/pv_rcnn/debug/ckpt/checkpoint_epoch_2.pth'
    # model_path = '../output/kitti/pv_rcnn_8369.pth'
    main(cfg_path, full_model_path, save_3d=True, tag=tag)