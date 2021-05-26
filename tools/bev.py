import argparse
import glob
from pathlib import Path

import mayavi.mlab as mlab
import numpy as np
import torch
import pickle

import pcdet
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from visual_utils import visualize_utils as V
import matplotlib.lines as lines
import matplotlib.pyplot as plt


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin',
                 info_path=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list
        self.range_config = dataset_cfg.get('RANGE_CONFIG', False)
        self.info_path = info_path
        if self.info_path is not None:
            with open(Path(self.info_path), 'rb') as f:
                self.infos = pickle.load(f)

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }
        # import pudb
        # pudb.set_trace()
        if self.info_path is not None:
            info = self.infos[index]

        if self.range_config:
            # data_dict = input_dict
            data_dict = self.prepare_data(data_dict=input_dict, process=False)
        else:
            data_dict = self.prepare_data(data_dict=input_dict)

        if self.range_config:
            beam_inclination_range = info.get('beam_inclination_range', (
            -0.43458698374658805, 0.03490658503988659)) if self.info_path is not None else (
            -0.43458698374658805, 0.03490658503988659)
            extrinsic = info.get('extrinsic', None) if self.info_path is not None else None
            data_dict.update({
                'beam_inclination_range': beam_inclination_range,
                'extrinsic': extrinsic,
                'range_image_shape': self.range_config.get('RANGE_IMAGE_SHAPE', [64, 2650]),
            })
            import pcdet.datasets.waymo.waymo_utils as waymo_utils
            data_dict = waymo_utils.convert_point_cloud_to_range_image(data_dict, self.training)
            data_dict['range_image'] = np.concatenate((data_dict['range_image'], data_dict['ri_xyz']), axis=0)
            data_dict.pop('ri_xyz', None)
            points_feature_num = data_dict['points'].shape[1]
            data_dict['points'] = np.concatenate((data_dict['points'], data_dict['ri_indices']), axis=1)
            data_dict = self.prepare_data(data_dict=data_dict, augment=False)
            data_dict['points'] = data_dict['points'][:, :points_feature_num]
            data_dict.pop('beam_inclination_range', None)
            data_dict.pop('extrinsic', None)
            data_dict.pop('range_image_shape', None)
        if self.info_path is not None:
            data_dict.update({'gt_boxes': info['annos']['gt_boxes_lidar']})

        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--info_path', type=str, default=None)

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def transform_to_img(xmin, xmax, ymin, ymax, pc_range, res=0.1):
    xmin_img = ymax / res - pc_range[1] / res
    xmax_img = ymin / res - pc_range[1] / res
    ymin_img = -xmax / res + pc_range[3] / res
    ymax_img = -xmin / res + pc_range[3] / res

    return xmin_img, xmax_img, ymin_img, ymax_img


def draw_boxes(ax, boxes, pc_range, color='green'):
    corners = V.boxes_to_corners_3d(boxes)
    # corners = corners.transpose((1, 2, 0))
    for o in range(len(boxes)):
        x1, x2, x3, x4 = corners[o][:4, 0]
        y1, y2, y3, y4 = corners[o][:4, 1]

        x1, x2, y1, y2 = transform_to_img(x1, x2, y1, y2, pc_range)
        x3, x4, y3, y4 = transform_to_img(x3, x4, y3, y4, pc_range)
        # ps = []
        # polygon = np.zeros([5, 2], dtype=np.float32)
        # polygon[0, 0] = x1
        # polygon[1, 0] = x2
        # polygon[2, 0] = x3
        # polygon[3, 0] = x4
        # polygon[4, 0] = x1
        #
        # polygon[0, 1] = y1
        # polygon[1, 1] = y2
        # polygon[2, 1] = y3
        # polygon[3, 1] = y4
        # polygon[4, 1] = y1

        line1 = [(x1, y1), (x2, y2)]
        line2 = [(x2, y2), (x3, y3)]
        line3 = [(x3, y3), (x4, y4)]
        line4 = [(x4, y4), (x1, y1)]
        (line1_xs, line1_ys) = zip(*line1)
        (line2_xs, line2_ys) = zip(*line2)
        (line3_xs, line3_ys) = zip(*line3)
        (line4_xs, line4_ys) = zip(*line4)
        ax.add_line(lines.Line2D(line1_xs, line1_ys, linewidth=1, color=color))
        ax.add_line(lines.Line2D(line2_xs, line2_ys, linewidth=1, color=color))
        ax.add_line(lines.Line2D(line3_xs, line3_ys, linewidth=1, color=color))
        ax.add_line(lines.Line2D(line4_xs, line4_ys, linewidth=1, color=color))


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger, info_path=args.info_path
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    pc_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
    res = 0.1
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            # import pudb
            # pudb.set_trace()
            mask = pred_dicts[0]['pred_boxes'][:, 3:5] < 20
            mask = mask.all(dim=1)
            mask2 = pred_dicts[0]['pred_boxes'][:, 4] / pred_dicts[0]['pred_boxes'][:, 3]
            mask2 = (mask2 < 20) & (mask2 > 0.05)
            mask2 = mask2 & mask

            points = data_dict['points'][:, 1:4].cpu().numpy()
            # import pudb
            # pudb.set_trace()
            x_points = points[:, 0]
            y_points = points[:, 1]
            # z_points = points[:, 2]
            # reflectance = points[:, 3]

            # INITIALIZE EMPTY ARRAY - of the dimensions we want
            x_max = int((pc_range[4] - pc_range[1]) / res)
            y_max = int((pc_range[3] - pc_range[0]) / res)
            # top = np.zeros([y_max+1, x_max+1, z_max+1], dtype=np.float32)
            top = np.ones([y_max + 1, x_max + 1], dtype=np.float32)

            # FILTER - To return only indices of points within desired cube
            # Three filters for: Front-to-back, side-to-side, and height ranges
            # Note left side is positive y axis in LIDAR coordinates
            f_filt = np.logical_and((x_points > pc_range[0]), (x_points < pc_range[3]))
            s_filt = np.logical_and((y_points > pc_range[1]), (y_points < pc_range[4]))
            filt = np.logical_and(f_filt, s_filt)
            indices = np.argwhere(filt).flatten()
            # import pudb
            # pudb.set_trace()
            xi_points = x_points[indices]
            yi_points = y_points[indices]
            # CONVERT TO PIXEL POSITION VALUES - Based on resolution
            x_img = (yi_points / res).astype(np.int32)  # x axis is -y in LIDAR
            y_img = (-xi_points / res).astype(np.int32)  # y axis is -x in LIDAR

            # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
            # floor & ceil used to prevent anything being rounded to below 0 after
            # shift
            x_img -= int(np.floor(pc_range[1] / res))
            y_img -= int(np.floor(pc_range[0] / res))
            # FILL PIXEL VALUES IN IMAGE ARRAY
            # top[y_img, x_img, i] = pixel_values
            top[y_img, x_img] = 0
            top = (top / np.max(top) * 255).astype(np.uint8)
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(top, aspect='equal', cmap='gray')

            draw_boxes(ax, data_dict.get('gt_boxes', None)[0].cpu().numpy(), pc_range, color='blue')
            # import pudb
            # pudb.set_trace()
            draw_boxes(ax, pred_dicts[0]['pred_boxes'][mask2][:100].cpu().numpy(), pc_range, color='red')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig('result/bev-%d-rrcnn.png' % idx)
            plt.clf()
            # rgba_colors = np.zeros((points.shape[0], 4))
            # rgba_colors[:, 2] = 1
            # rgba_colors[:, 3] = points[:, 2]
            # plt.scatter(points[:, 0], points[:, 1], s=0.5, color=rgba_colors[:, :3])
            #
            # gt_boxes = data_dict.get('gt_boxes', None)[0].cpu().numpy()
            # gt_corners = V.boxes_to_corners_3d(gt_boxes)
            # gt_corners = gt_corners.transpose((1, 2, 0))
            # x1, x2, x3, x4 = gt_corners[:4, 0]
            # y1, y2, y3, y4 = gt_corners[:4, 1]
            # plt.plot((x1, x2, x3, x4, x1), (y1, y2, y3, y4, y1), color='yellowgreen', linewidth=2)
            # pred_boxes = pred_dicts[0]['pred_boxes'][mask2].cpu().numpy()
            # pred_corners = V.boxes_to_corners_3d(pred_boxes[:100])
            # pred_corners = pred_corners.transpose((1, 2, 0))
            # x1, x2, x3, x4 = pred_corners[:4, 0]
            # y1, y2, y3, y4 = pred_corners[:4, 1]
            # plt.plot((x1, x2, x3, x4, x1), (y1, y2, y3, y4, y1), color='red', linewidth=2)
            # plt.savefig('bev-%d-rsn.png'%idx)
            # plt.clf()
            # plt.show()

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
