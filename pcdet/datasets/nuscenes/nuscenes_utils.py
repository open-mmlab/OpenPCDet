"""
The NuScenes data pre-processing is borrowed from
https://github.com/traveller59/second.pytorch and https://github.com/poodarchu/Det3D
"""

from pathlib import Path
import tqdm
import numpy as np
from functools import reduce
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion


map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}


def get_available_scenes(nusc):
    available_scenes = []
    print('total scene num:', len(nusc.scene))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            if not Path(lidar_path).exists():
                scene_not_exist = True
                break
            else:
                break
            # if not sd_rec['next'] == '':
            #     sd_rec = nusc.get('sample_data', sd_rec['next'])
            # else:
            #     has_more_frames = False
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num:', len(available_scenes))
    return available_scenes


def get_sample_data(nusc, sample_data_token, selected_anntokens=None):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    Args:
        nusc:
        sample_data_token: Sample_data token.
        selected_anntokens: If provided only return the selected annotation.

    Returns:

    """
    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = None
        imsize = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    if selected_anntokens is not None:
        boxes = list(map(nusc.get_box, selected_anntokens))
    else:
        boxes = nusc.get_boxes(sample_data_token)

    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)

        #  Move box to sensor coord system
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        box_list.append(box)

    return data_path, box_list, cam_intrinsic


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw


def fill_trainval_infos(data_path, nusc, train_scenes, val_scenes, test=False, max_sweeps=10):
    train_nusc_infos = []
    val_nusc_infos = []
    progress_bar = tqdm.tqdm(total=len(nusc.sample), desc='create_info', dynamic_ncols=True)

    ref_chan = 'LIDAR_TOP'  # The radar channel from which we track back n sweeps to aggregate the point cloud.
    chan = 'LIDAR_TOP'  # The reference channel of the current sample_rec that the point clouds are mapped to.

    for index, sample in enumerate(nusc.sample):
        progress_bar.update()

        ref_sd_token = sample['data'][ref_chan]
        ref_sd_rec = nusc.get('sample_data', ref_sd_token)
        ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        ref_lidar_path, ref_boxes, _ = get_sample_data(nusc, ref_sd_token)

        ref_cam_front_token = sample['data']['CAM_FRONT']
        ref_cam_path, _, ref_cam_intrinsic = nusc.get_sample_data(ref_cam_front_token)

        # Homogeneous transform from ego car frame to reference frame
        ref_from_car = transform_matrix(
            ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True
        )

        # Homogeneous transformation matrix from global to _current_ ego car frame
        car_from_global = transform_matrix(
            ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']), inverse=True,
        )

        info = {
            'lidar_path': Path(ref_lidar_path).relative_to(data_path).__str__(),
            'cam_front_path': Path(ref_cam_path).relative_to(data_path).__str__(),
            'cam_intrinsic': ref_cam_intrinsic,
            'token': sample['token'],
            'sweeps': [],
            'ref_from_car': ref_from_car,
            'car_from_global': car_from_global,
            'timestamp': ref_time,
        }

        sample_data_token = sample['data'][chan]
        curr_sd_rec = nusc.get('sample_data', sample_data_token)
        sweeps = []
        while len(sweeps) < max_sweeps - 1:
            if curr_sd_rec['prev'] == '':
                if len(sweeps) == 0:
                    sweep = {
                        'lidar_path': ref_lidar_path,
                        'sample_data_token': curr_sd_rec['token'],
                        'transform_matrix': None,
                        'time_lag': curr_sd_rec['timestamp'] * 0,
                    }
                    sweeps.append(sweep)
                else:
                    sweeps.append(sweeps[-1])
            else:
                curr_sd_rec = nusc.get('sample_data', curr_sd_rec['prev'])

                # Get past pose
                current_pose_rec = nusc.get('ego_pose', curr_sd_rec['ego_pose_token'])
                global_from_car = transform_matrix(
                    current_pose_rec['translation'], Quaternion(current_pose_rec['rotation']), inverse=False,
                )

                # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
                current_cs_rec = nusc.get(
                    'calibrated_sensor', curr_sd_rec['calibrated_sensor_token']
                )
                car_from_current = transform_matrix(
                    current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']), inverse=False,
                )

                tm = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])

                lidar_path = nusc.get_sample_data_path(curr_sd_rec['token'])

                time_lag = ref_time - 1e-6 * curr_sd_rec['timestamp']

                sweep = {
                    'lidar_path': Path(lidar_path).relative_to(data_path).__str__(),
                    'sample_data_token': curr_sd_rec['token'],
                    'transform_matrix': tm,
                    'global_from_car': global_from_car,
                    'car_from_current': car_from_current,
                    'time_lag': time_lag,
                }
                sweeps.append(sweep)

        info['sweeps'] = sweeps

        assert len(info['sweeps']) == max_sweeps - 1, \
            f"sweep {curr_sd_rec['token']} only has {len(info['sweeps'])} sweeps, " \
            f"you should duplicate to sweep num {max_sweeps - 1}"

        if not test:
            annotations = [nusc.get('sample_annotation', token) for token in sample['anns']]

            # the filtering gives 0.5~1 map improvement
            mask = np.array([(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
                             for anno in annotations], dtype=bool).reshape(-1)

            locs = np.array([b.center for b in ref_boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in ref_boxes]).reshape(-1, 3)[:, [1, 0, 2]]  # wlh == > dxdydz (lwh)
            velocity = np.array([b.velocity for b in ref_boxes]).reshape(-1, 3)
            rots = np.array([quaternion_yaw(b.orientation) for b in ref_boxes]).reshape(-1, 1)
            names = np.array([b.name for b in ref_boxes])
            tokens = np.array([b.token for b in ref_boxes])
            gt_boxes = np.concatenate([locs, dims, rots, velocity[:, :2]], axis=1)

            assert len(annotations) == len(gt_boxes) == len(velocity)

            info['gt_boxes'] = gt_boxes[mask, :]
            info['gt_boxes_velocity'] = velocity[mask, :]
            info['gt_names'] = np.array([map_name_from_general_to_detection[name] for name in names])[mask]
            info['gt_boxes_token'] = tokens[mask]

        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    progress_bar.close()
    return train_nusc_infos, val_nusc_infos
