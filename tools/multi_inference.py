import rospy
import ros_numpy
import numpy as np
import os
import sys
import torch
import time 
import glob
from pathlib import Path

from std_msgs.msg import Header
from pyquaternion import Quaternion
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray

from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils

import cupy as cp
from collections import deque
from copy import deepcopy
from functools import reduce


class DemoDataset(DatasetTemplate):
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

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def yaw2quaternion(yaw: float) -> Quaternion:
    return Quaternion(axis=[0,0,1], radians=yaw)

def transform_matrix(translation: np.ndarray = np.array([0, 0, 0]),
                     rotation: Quaternion = Quaternion([1, 0, 0, 0]),
                     inverse: bool = False) -> np.ndarray:
    """
    Convert pose to transformation matrix.
    :param translation: <np.float32: 3>. Translation in x, y, z.
    :param rotation: Rotation in quaternions (w ri rj rk).
    :param inverse: Whether to compute inverse transform matrix.
    :return: <np.float32: 4, 4>. Transformation matrix.
    """
    tm = np.eye(4)
    if inverse:
        rot_inv = rotation.rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix
        tm[:3, 3] = np.transpose(np.array(translation))
    return tm

def get_annotations_indices(types, thresh, label_preds, scores):
    indexs = []
    annotation_indices = []
    for i in range(label_preds.shape[0]):
        if label_preds[i] == types:
            indexs.append(i)
    for index in indexs:
        if scores[index] >= thresh:
            annotation_indices.append(index)
    return annotation_indices  


def remove_low_score_nu(image_anno, thresh):
    img_filtered_annotations = {}
    label_preds_ = image_anno["pred_labels"].detach().cpu().numpy()
    scores_ = image_anno["pred_scores"].detach().cpu().numpy()
    
    car_indices =                  get_annotations_indices(1, 0.45, label_preds_, scores_)
    truck_indices =                get_annotations_indices(2, 0.45, label_preds_, scores_)
    construction_vehicle_indices = get_annotations_indices(3, 0.45, label_preds_, scores_)
    bus_indices =                  get_annotations_indices(4, 0.35, label_preds_, scores_)
    trailer_indices =              get_annotations_indices(5, 0.4, label_preds_, scores_)
    barrier_indices =              get_annotations_indices(6, 0.4, label_preds_, scores_)
    motorcycle_indices =           get_annotations_indices(7, 0.15, label_preds_, scores_)
    bicycle_indices =              get_annotations_indices(8, 0.15, label_preds_, scores_)
    pedestrain_indices =           get_annotations_indices(9, 0.10, label_preds_, scores_)
    traffic_cone_indices =         get_annotations_indices(10, 0.1, label_preds_, scores_)
    
    for key in image_anno.keys():
        if key == 'metadata':
            continue
        img_filtered_annotations[key] = (
            image_anno[key][car_indices +
                            pedestrain_indices + 
                            bicycle_indices +
                            bus_indices +
                            construction_vehicle_indices +
                            traffic_cone_indices +
                            trailer_indices +
                            barrier_indices +
                            truck_indices
                            ])

    return img_filtered_annotations


class Processor_ROS:
    def __init__(self, config_path, model_path):
        self.points = None
        self.config_path = config_path
        self.model_path = model_path
        self.device = None
        self.net = None

        self.lidar_deque = deque(maxlen=5)
        self.current_frame = {
            "lidar_stamp": None,
            "lidar_seq": None,
            "points": None,
            "odom_seq": None,
            "odom_stamp": None,
            "translation": None,
            "rotation": None
        }
        self.pc_list = deque(maxlen=5)
        self.inputs = None

        
    def initialize(self):
        self.read_config()
        
    def read_config(self):
        config_path = self.config_path
        cfg_from_yaml_file(self.config_path, cfg)
        self.logger = common_utils.create_logger()
        self.demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=Path("/home/muzi2045/Documents/project/OpenPCDet/data/kitti/velodyne/000001.bin"),
            ext='.bin')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.demo_dataset)
        self.net.load_params_from_file(filename=self.model_path, logger=self.logger, to_cpu=True)
        self.net = self.net.to(self.device).eval()

         # nuscenes dataset
        lidar2imu_t = np.array([0.985793, 0.0, 1.84019])
        lidar2imu_r = Quaternion([0.706749235, -0.01530099378, 0.0173974518, -0.7070846])
        self.lidar2imu = transform_matrix(lidar2imu_t, lidar2imu_r, inverse=True)
        self.imu2lidar = transform_matrix(lidar2imu_t, lidar2imu_r, inverse=False)

    def run(self):
        t_t = time.time()
        input_dict = {
            'points': self.points,
            'frame_id': 0,
        }

        data_dict = self.demo_dataset.prepare_data(data_dict=input_dict)
        data_dict = self.demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)

        torch.cuda.synchronize()
        t = time.time()

        pred_dicts, _ = self.net.forward(data_dict)
        
        torch.cuda.synchronize()
        print(f" pvrcnn inference cost time: {time.time() - t}")

        # pred = remove_low_score_nu(pred_dicts[0], 0.45)
        # boxes_lidar = pred["pred_boxes"].detach().cpu().numpy()
        # scores = pred["pred_scores"].detach().cpu().numpy()
        # types = pred["pred_labels"].detach().cpu().numpy()

        boxes_lidar = pred_dicts[0]["pred_boxes"].detach().cpu().numpy()
        scores = pred_dicts[0]["pred_scores"].detach().cpu().numpy()
        types = pred_dicts[0]["pred_labels"].detach().cpu().numpy()

        # print(f" pred boxes: { boxes_lidar }")
        # print(f" pred_scores: { scores }")
        # print(f" pred_labels: { types }")

        return scores, boxes_lidar, types

    def get_lidar_data(self, input_points: dict):
        print("get one frame lidar data.")
        self.current_frame["lidar_stamp"] = input_points['stamp']
        self.current_frame["lidar_seq"] = input_points['seq']
        self.current_frame["points"] = input_points['points'].T   
        self.lidar_deque.append(deepcopy(self.current_frame))
        if len(self.lidar_deque) == 5:

            ref_from_car = self.imu2lidar
            car_from_global = transform_matrix(self.lidar_deque[-1]['translation'], self.lidar_deque[-1]['rotation'], inverse=True)

            ref_from_car_gpu = cp.asarray(ref_from_car)
            car_from_global_gpu = cp.asarray(car_from_global)

            for i in range(len(self.lidar_deque) - 1):
                last_pc = self.lidar_deque[i]['points']
                last_pc_gpu = cp.asarray(last_pc)

                global_from_car = transform_matrix(self.lidar_deque[i]['translation'], self.lidar_deque[i]['rotation'], inverse=False)
                car_from_current = self.lidar2imu
                global_from_car_gpu = cp.asarray(global_from_car)
                car_from_current_gpu = cp.asarray(car_from_current)

                transform = reduce(
                    cp.dot,
                    [ref_from_car_gpu, car_from_global_gpu, global_from_car_gpu, car_from_current_gpu],
                )
                last_pc_gpu = cp.vstack((last_pc_gpu[:3, :], cp.ones(last_pc_gpu.shape[1])))
                last_pc_gpu = cp.dot(transform, last_pc_gpu)

                self.pc_list.append(last_pc_gpu[:3, :])

            current_pc = self.lidar_deque[-1]['points']
            current_pc_gpu = cp.asarray(current_pc)
            self.pc_list.append(current_pc_gpu[:3,:])

            all_pc = np.zeros((5, 0), dtype=float)
            for i in range(len(self.pc_list)):
                tmp_pc = cp.vstack((self.pc_list[i], cp.zeros((2, self.pc_list[i].shape[1]))))
                tmp_pc = cp.asnumpy(tmp_pc)
                ref_timestamp = self.lidar_deque[-1]['lidar_stamp'].to_sec()
                timestamp = self.lidar_deque[i]['lidar_stamp'].to_sec()
                tmp_pc[3, ...] = self.lidar_deque[i]['points'][3, ...]
                tmp_pc[4, ...] = ref_timestamp - timestamp
                all_pc = np.hstack((all_pc, tmp_pc))
            
            all_pc = all_pc.T
            print(f" concate pointcloud shape: {all_pc.shape}")

            self.points = all_pc
            sync_cloud = xyz_array_to_pointcloud2(all_pc[:, :3], stamp=self.lidar_deque[-1]["lidar_stamp"], frame_id="lidar_top")
            pub_sync_cloud.publish(sync_cloud)
            
            return True

    def get_odom_data(self, input_odom):
        self.current_frame["odom_stamp"] = input_odom.header.stamp
        self.current_frame["odom_seq"] = input_odom.header.seq
        x_t = input_odom.pose.pose.position.x
        y_t = input_odom.pose.pose.position.y
        z_t = input_odom.pose.pose.position.z
        self.current_frame["translation"] = np.array([x_t, y_t, z_t])
        x_r = input_odom.pose.pose.orientation.x
        y_r = input_odom.pose.pose.orientation.y
        z_r = input_odom.pose.pose.orientation.z
        w_r = input_odom.pose.pose.orientation.w
        self.current_frame["rotation"] = Quaternion([w_r, x_r, y_r, z_r])

def get_xyz_points(cloud_array, remove_nans=True, dtype=np.float):
    '''
    '''
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]

    points = np.zeros(cloud_array.shape + (5,), dtype=dtype)
    points[...,0] = cloud_array['x']
    points[...,1] = cloud_array['y']
    points[...,2] = cloud_array['z']
    return points

def xyz_array_to_pointcloud2(points_sum, stamp=None, frame_id=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array of points.
    '''
    msg = PointCloud2()
    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = points_sum.shape[0]
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)
        # PointField('i', 12, PointField.FLOAT32, 1)
        ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = points_sum.shape[0]
    msg.is_dense = int(np.isfinite(points_sum).all())
    msg.data = np.asarray(points_sum, np.float32).tostring()
    return msg

def rslidar_callback(msg):
    t_t = time.time()
    arr_bbox = BoundingBoxArray()

    msg_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
    np_p = get_xyz_points(msg_cloud, True)
    print("  ")
    seq = msg.header.seq
    stamp = msg.header.stamp
    
    input_points = {
        'stamp': stamp,
        'seq': seq,
        'points': np_p
    }

    if(proc_1.get_lidar_data(input_points)):

        scores, dt_box_lidar, types = proc_1.run()
        if scores.size != 0:
            for i in range(scores.size):
                bbox = BoundingBox()
                bbox.header.frame_id = msg.header.frame_id
                bbox.header.stamp = rospy.Time.now()
                q = yaw2quaternion(float(dt_box_lidar[i][6]))
                bbox.pose.orientation.x = q[1]
                bbox.pose.orientation.y = q[2]
                bbox.pose.orientation.z = q[3]
                bbox.pose.orientation.w = q[0]           
                bbox.pose.position.x = float(dt_box_lidar[i][0])
                bbox.pose.position.y = float(dt_box_lidar[i][1])
                bbox.pose.position.z = float(dt_box_lidar[i][2])
                bbox.dimensions.x = float(dt_box_lidar[i][3])
                bbox.dimensions.y = float(dt_box_lidar[i][4])
                bbox.dimensions.z = float(dt_box_lidar[i][5])
                bbox.value = scores[i]
                bbox.label = int(types[i])
                arr_bbox.boxes.append(bbox)
        print("total callback time: ", time.time() - t_t)
        arr_bbox.header.frame_id = msg.header.frame_id
        arr_bbox.header.stamp = msg.header.stamp
        if len(arr_bbox.boxes) is not 0:
            pub_arr_bbox.publish(arr_bbox)
            arr_bbox.boxes = []
        else:
            arr_bbox.boxes = []
            pub_arr_bbox.publish(arr_bbox)


def odom_callback(msg):
    '''
    get odom data
    '''
    proc_1.get_odom_data(msg)
   
if __name__ == "__main__":

    global proc
    ## PVRCNN
    # config_path = '/home/muzi2045/Documents/project/OpenPCDet/tools/cfgs/kitti_models/pv_rcnn.yaml'
    # model_path = '/home/muzi2045/Documents/project/OpenPCDet/data/model/pv_rcnn_8369.pth'

    ## PointRCNN
    # config_path = '/home/muzi2045/Documents/project/OpenPCDet/tools/cfgs/kitti_models/pointrcnn.yaml'
    # model_path = '/home/muzi2045/Documents/project/OpenPCDet/data/model/pointrcnn_7870.pth'

    ## PartA2_free
    # config_path = '/home/muzi2045/Documents/project/OpenPCDet/tools/cfgs/kitti_models/PartA2_free.yaml'
    # model_path = '/home/muzi2045/Documents/project/OpenPCDet/data/model/PartA2_free_7872.pth'

    ## PointPillar
    # config_path = '/home/muzi2045/Documents/project/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml'
    # model_path = '/home/muzi2045/Documents/project/OpenPCDet/data/model/pointpillar_7728.pth'

    ## SECOND
    # config_path = '/home/muzi2045/Documents/project/OpenPCDet/tools/cfgs/kitti_models/second.yaml'
    # model_path = '/home/muzi2045/Documents/project/OpenPCDet/data/model/second_7862.pth'


    ## SECOND_MultiHead (trained on nuscenes dataset)
    # config_path = '/home/muzi2045/Documents/project/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_second_multihead.yaml'
    # model_path = '/home/muzi2045/Documents/project/OpenPCDet/data/model/cbgs_second_multihead_nds6229.pth'

    ## PP_MutliHead (trained on nuscenes dataset)
    config_path = '/home/muzi2045/Documents/project/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_pp_multihead.yaml'
    model_path = '/home/muzi2045/Documents/project/OpenPCDet/data/model/pp_multihead_nds5823.pth'

    proc_1 = Processor_ROS(config_path, model_path)
    
    proc_1.initialize()
    
    rospy.init_node('pcdet_ros_node')
    sub_lidar_topic = [ "/velodyne_points", 
                        "/top/rslidar_points",
                        "/points_raw", 
                        "/lidar_protector/merged_cloud", 
                        "/merged_cloud",
                        "/lidar_top", 
                        "/roi_pclouds",
                        "/livox/lidar",
                        "/SimOneSM_PointCloud_0"]
    
    sub_ = rospy.Subscriber(sub_lidar_topic[5], PointCloud2, rslidar_callback, queue_size=1, buff_size=2**24)

    sub_odom_topic = ["/golfcar/odom",
                      "/aligned/odometry",
                      "/odom"]
    sub_odom = rospy.Subscriber(
        sub_odom_topic[2], Odometry, odom_callback, queue_size=10, buff_size=2**10, tcp_nodelay=True)
    
    pub_arr_bbox = rospy.Publisher("pp_boxes", BoundingBoxArray, queue_size=1)
    pub_sync_cloud = rospy.Publisher("sync_5sweeps_cloud", PointCloud2, queue_size=1)

    print("[+] PCDet ros_node has started!")    
    rospy.spin()