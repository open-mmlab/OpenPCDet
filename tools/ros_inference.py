#!/usr/bin/env python3
# ROS
import rospy
import rospkg
import ros_numpy

import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

import sys
sys.path.append('/OpenPCDet')
print (sys.path)

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.models import build_ros_network


class LidarDetector(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
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

        self.sub_lidar = rospy.Subscriber("/os_cloud_node/points", PointCloud2, self.lidar_callback, queue_size=1, buff_size=2**24) # "/kitti/velo/pointcloud"
        self.scan_received = False
        self.point_xyzi_arr = None

    def lidar_callback(self, msg):
        self.is_callback = True
        # self.scan = msg
        self.on_scan(msg)

    def on_scan(self, scan):
        # if (scan is None or not self.is_callback):
        #     return
        if (scan is None):
            return

        gen = []
        # msg_numpy = ros_numpy.numpify(scan)
        self.point_xyzi_arr = self.get_xyzi_points(ros_numpy.point_cloud2.pointcloud2_to_array(scan), remove_nans=True)
        if (len(self.point_xyzi_arr) == 0):
            return
        # rospy.loginfo("Got scan %d points", len(self.point_xyzi_arr))
        # print(self.point_xyzi_arr)
        self.scan_received = True


    def get_xyzi_points(self, cloud_array, remove_nans=True, dtype=np.float):
        '''Pulls out x, y, and z columns from the cloud recordarray, and returns
            a 3xN matrix.
        '''
        # remove crap points
        # print(cloud_array.dtype.names)
        if remove_nans:
            mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(
                cloud_array['z']) & np.isfinite(cloud_array['intensity'])
            cloud_array = cloud_array[mask]

        # pull out x, y, and z values + intensity
        points = np.zeros(cloud_array.shape + (4,), dtype=dtype)
        points[..., 0] = cloud_array['x']
        points[..., 1] = cloud_array['y']
        points[..., 2] = cloud_array['z']
        points[..., 3] = cloud_array['intensity']

        return points


    def __len__(self):
        return len(self.sample_file_list)

    #TODO put this value as input
    def getitem(self):
        if self.scan_received:
            points = self.point_xyzi_arr.reshape(-1, 4)
        else:
            raise NotImplementedError

        print()
        input_dict = {
            'points': points,
            'frame_id': 0,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    rospy.init_node('Lidar_Object_Detection', anonymous=True)
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    
    demo_dataset = LidarDetector(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')
    model_loaded = False
    if(model_loaded == False):
        model = build_ros_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    model_loaded = True

    r = rospy.Rate(10)  # 10hz
    while not rospy.is_shutdown():
        with torch.no_grad():
        #     for idx, data_dict in enumerate(demo_dataset):
                # logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.getitem()
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            # new_pred_dict = pred_dicts[0]['pred_boxes'][pred_dicts[0]['pred_boxes'][:,0] < 5.0]

            print(pred_dicts[0]['pred_boxes'])
            # print(new_pred_dict)
            # print(data_dict['points'][:, 1:])
            
            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            # )

        r.sleep()

    # logger.info('Demo done.')
    rospy.spin()

if __name__ == '__main__':
    main()
