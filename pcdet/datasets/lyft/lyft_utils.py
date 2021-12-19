"""
The Lyft data pre-processing and evaluation is modified from
https://github.com/poodarchu/Det3D
"""

import operator
from functools import reduce
from pathlib import Path

import numpy as np
import tqdm
from lyft_dataset_sdk.utils.data_classes import Box, Quaternion
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.geometry_utils import transform_matrix
from lyft_dataset_sdk.eval.detection.mAP_evaluation import Box3D


def get_available_scenes(lyft):
    available_scenes = []
    print('total scene num:', len(lyft.scene))
    for scene in lyft.scene:
        scene_token = scene['token']
        scene_rec = lyft.get('scene', scene_token)
        sample_rec = lyft.get('sample', scene_rec['first_sample_token'])
        sd_rec = lyft.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = lyft.get_sample_data(sd_rec['token'])
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


def get_sample_data(lyft, sample_data_token):
    sd_rec = lyft.get("sample_data", sample_data_token)
    cs_rec = lyft.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])

    sensor_rec = lyft.get("sensor", cs_rec["sensor_token"])
    pose_rec = lyft.get("ego_pose", sd_rec["ego_pose_token"])

    boxes = lyft.get_boxes(sample_data_token)

    box_list = []
    for box in boxes:
        box.translate(-np.array(pose_rec["translation"]))
        box.rotate(Quaternion(pose_rec["rotation"]).inverse)

        box.translate(-np.array(cs_rec["translation"]))
        box.rotate(Quaternion(cs_rec["rotation"]).inverse)

        box_list.append(box)

    return box_list, pose_rec


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


def fill_trainval_infos(data_path, lyft, train_scenes, val_scenes, test=False, max_sweeps=10):
    train_lyft_infos = []
    val_lyft_infos = []
    progress_bar = tqdm.tqdm(total=len(lyft.sample), desc='create_info', dynamic_ncols=True)

    # ref_chans = ["LIDAR_TOP", "LIDAR_FRONT_LEFT", "LIDAR_FRONT_RIGHT"]
    ref_chan = "LIDAR_TOP"

    for index, sample in enumerate(lyft.sample):
        progress_bar.update()

        ref_info = {}
        ref_sd_token = sample["data"][ref_chan]
        ref_sd_rec = lyft.get("sample_data", ref_sd_token)
        ref_cs_token = ref_sd_rec["calibrated_sensor_token"]
        ref_cs_rec = lyft.get("calibrated_sensor", ref_cs_token)

        ref_to_car = transform_matrix(
            ref_cs_rec["translation"],
            Quaternion(ref_cs_rec["rotation"]),
            inverse=False,
        )

        ref_from_car = transform_matrix(
            ref_cs_rec["translation"],
            Quaternion(ref_cs_rec["rotation"]),
            inverse=True,
        )

        ref_lidar_path = lyft.get_sample_data_path(ref_sd_token)

        ref_boxes, ref_pose_rec = get_sample_data(lyft, ref_sd_token)
        ref_time = 1e-6 * ref_sd_rec["timestamp"]
        car_from_global = transform_matrix(
            ref_pose_rec["translation"],
            Quaternion(ref_pose_rec["rotation"]),
            inverse=True,
        )

        car_to_global = transform_matrix(
            ref_pose_rec["translation"],
            Quaternion(ref_pose_rec["rotation"]),
            inverse=False,
        )

        info = {
            "lidar_path": Path(ref_lidar_path).relative_to(data_path).__str__(),
            "ref_from_car": ref_from_car,
            "ref_to_car": ref_to_car,
            'token': sample['token'],
            'car_from_global': car_from_global,
            'car_to_global': car_to_global,
            'timestamp': ref_time,
            'sweeps': []
        }

        sample_data_token = sample['data'][ref_chan]
        curr_sd_rec = lyft.get('sample_data', sample_data_token)
        sweeps = []

        while len(sweeps) < max_sweeps - 1:
            if curr_sd_rec['prev'] == '':
                if len(sweeps) == 0:
                    sweep = {
                        'lidar_path': Path(ref_lidar_path).relative_to(data_path).__str__(),
                        'sample_data_token': curr_sd_rec['token'],
                        'transform_matrix': None,
                        'time_lag': curr_sd_rec['timestamp'] * 0,
                    }
                    sweeps.append(sweep)
                else:
                    sweeps.append(sweeps[-1])
            else:
                curr_sd_rec = lyft.get('sample_data', curr_sd_rec['prev'])

                # Get past pose
                current_pose_rec = lyft.get('ego_pose', curr_sd_rec['ego_pose_token'])
                global_from_car = transform_matrix(
                    current_pose_rec['translation'], Quaternion(current_pose_rec['rotation']), inverse=False,
                )

                # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
                current_cs_rec = lyft.get(
                    'calibrated_sensor', curr_sd_rec['calibrated_sensor_token']
                )
                car_from_current = transform_matrix(
                    current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']), inverse=False,
                )

                tm = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])

                lidar_path = lyft.get_sample_data_path(curr_sd_rec['token'])

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

        if not test:
            annotations = [
                lyft.get("sample_annotation", token) for token in sample["anns"]
            ]

            locs = np.array([b.center for b in ref_boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in ref_boxes]).reshape(-1, 3)[:, [1, 0, 2]]
            rots = np.array([quaternion_yaw(b.orientation) for b in ref_boxes]).reshape(
                -1, 1
            )
            velocity = np.array([b.velocity for b in ref_boxes]).reshape(-1, 3)
            names = np.array([b.name for b in ref_boxes])
            tokens = np.array([b.token for b in ref_boxes]).reshape(-1, 1)
            gt_boxes = np.concatenate([locs, dims, rots], axis=1)

            assert len(annotations) == len(gt_boxes)

            info["gt_boxes"] = gt_boxes
            info["gt_boxes_velocity"] = velocity
            info["gt_names"] = names
            info["gt_boxes_token"] = tokens

        if sample["scene_token"] in train_scenes:
            train_lyft_infos.append(info)
        else:
            val_lyft_infos.append(info)

    progress_bar.close()
    return train_lyft_infos, val_lyft_infos

def boxes_lidar_to_lyft(boxes3d, scores=None, labels=None):
    box_list = []
    for k in range(boxes3d.shape[0]):
        quat = Quaternion(axis=[0, 0, 1], radians=boxes3d[k, 6])
        box = Box(
            boxes3d[k, :3],
            boxes3d[k, [4, 3, 5]],  # wlh
            quat, label=labels[k] if labels is not None else np.nan,
            score=scores[k] if scores is not None else np.nan,
        )
        box_list.append(box)
    return box_list


def lidar_lyft_box_to_global(lyft, boxes, sample_token):
    s_record = lyft.get('sample', sample_token)
    sample_data_token = s_record['data']['LIDAR_TOP']

    sd_record = lyft.get('sample_data', sample_data_token)
    cs_record = lyft.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = lyft.get('sensor', cs_record['sensor_token'])
    pose_record = lyft.get('ego_pose', sd_record['ego_pose_token'])

    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(Quaternion(cs_record['rotation']))
        box.translate(np.array(cs_record['translation']))
        # Move box to global coord system
        box.rotate(Quaternion(pose_record['rotation']))
        box.translate(np.array(pose_record['translation']))
        box_list.append(box)
    return box_list


def convert_det_to_lyft_format(lyft, det_annos):
    sample_tokens = []
    det_lyft_box = [] 
    for anno in det_annos:
        sample_tokens.append(anno['metadata']['token'])

        boxes_lyft_list = boxes_lidar_to_lyft(anno['boxes_lidar'], anno['score'], anno['pred_labels'])
        boxes_list = lidar_lyft_box_to_global(lyft, boxes_lyft_list, anno['metadata']['token'])

        for idx, box in enumerate(boxes_list):
            name = anno['name'][idx]
            box3d = {
                'sample_token': anno['metadata']['token'],
                'translation': box.center.tolist(),
                'size': box.wlh.tolist(),
                'rotation': box.orientation.elements.tolist(),
                'name': name,
                'score': box.score
            }
            det_lyft_box.append(box3d)
    
    return det_lyft_box, sample_tokens


def load_lyft_gt_by_tokens(lyft, sample_tokens):
    """
    Modify from Lyft tutorial
    """

    gt_box3ds = []

    # Load annotations and filter predictions and annotations.
    for sample_token in sample_tokens:
        
        sample = lyft.get('sample', sample_token)

        sample_annotation_tokens = sample['anns']

        sample_lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = lyft.get("sample_data", sample_lidar_token)
        ego_pose = lyft.get("ego_pose", lidar_data["ego_pose_token"])
        ego_translation = np.array(ego_pose['translation'])
        
        for sample_annotation_token in sample_annotation_tokens:
            sample_annotation = lyft.get('sample_annotation', sample_annotation_token)
            sample_annotation_translation = sample_annotation['translation']
            
            class_name = sample_annotation['category_name']
            
            box3d = {
                'sample_token': sample_token,
                'translation': sample_annotation_translation,
                'size': sample_annotation['size'],
                'rotation': sample_annotation['rotation'],
                'name': class_name
            }
            gt_box3ds.append(box3d)
            
    return gt_box3ds


def format_lyft_results(classwise_ap, class_names, iou_threshold_list, version='trainval'):
    ret_dict = {}
    result = '----------------Lyft %s results-----------------\n' % version
    result += 'Average precision over IoUs: {}\n'.format(str(iou_threshold_list))
    for c_idx, class_name in enumerate(class_names):
        result += '{:<20}: \t {:.4f}\n'.format(class_name, classwise_ap[c_idx])
        ret_dict[class_name] = classwise_ap[c_idx]

    result += '--------------average performance-------------\n'
    mAP = np.mean(classwise_ap)
    result += 'mAP:\t {:.4f}\n'.format(mAP)

    ret_dict['mAP'] = mAP
    return result, ret_dict
