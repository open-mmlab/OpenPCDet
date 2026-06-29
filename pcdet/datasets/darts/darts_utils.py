import json
import operator
from functools import reduce
from pathlib import Path
from collections import defaultdict
import numpy as np
import tqdm
from pyquaternion import Quaternion
from typing import List, Tuple
import copy
class Box:
    """ Simple data class representing a 3d box including, label and score. """

    def __init__(self,
                 center: List[float],
                 size: List[float],
                 orientation: Quaternion,
                 label: int = np.nan,
                 score: float = np.nan,
                 name: str = None,
                 token: str = None):
        """
        :param center: Center of box given as x, y, z.
        :param size: Size of box in width, length, height.
        :param orientation: Box orientation.
        :param label: Integer label, optional.
        :param score: Classification score, optional.
        :param name: Box name, optional. Can be used e.g. for denote category name.
        :param token: Unique string identifier from DB.
        """
        assert not np.any(np.isnan(center))
        assert not np.any(np.isnan(size))
        assert len(center) == 3
        assert len(size) == 3
        assert type(orientation) == Quaternion

        self.center = np.array(center)
        self.wlh = np.array(size)
        self.orientation = orientation
        self.label = int(label) if not np.isnan(label) else label
        self.score = float(score) if not np.isnan(score) else score
        self.name = name
        self.token = token

    def __eq__(self, other):
        center = np.allclose(self.center, other.center)
        wlh = np.allclose(self.wlh, other.wlh)
        orientation = np.allclose(self.orientation.elements, other.orientation.elements)
        label = (self.label == other.label) or (np.isnan(self.label) and np.isnan(other.label))
        score = (self.score == other.score) or (np.isnan(self.score) and np.isnan(other.score))

        return center and wlh and orientation and label and score

    def __repr__(self):
        repr_str = 'label: {}, score: {:.2f}, xyz: [{:.2f}, {:.2f}, {:.2f}], wlh: [{:.2f}, {:.2f}, {:.2f}], ' \
                   'rot axis: [{:.2f}, {:.2f}, {:.2f}], ang(degrees): {:.2f}, ang(rad): {:.2f}, ' \
                   'name: {}, token: {}'

        return repr_str.format(self.label, self.score, self.center[0], self.center[1], self.center[2], self.wlh[0],
                               self.wlh[1], self.wlh[2], self.orientation.axis[0], self.orientation.axis[1],
                               self.orientation.axis[2], self.orientation.degrees, self.orientation.radians,
                               self.name, self.token)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """
        Return a rotation matrix.
        :return: <np.float: 3, 3>. The box's rotation matrix.
        """
        return self.orientation.rotation_matrix

    def translate(self, x: np.ndarray) -> None:
        """
        Applies a translation.
        :param x: <np.float: 3, 1>. Translation in x, y, z direction.
        """
        self.center += x

    def rotate(self, quaternion: Quaternion) -> None:
        """
        Rotates box.
        :param quaternion: Rotation to apply.
        """
        self.center = np.dot(quaternion.rotation_matrix, self.center)
        self.orientation = quaternion * self.orientation

class DARTS:
    def __init__(self, dataroot, version) -> None:
        self._root = Path(dataroot)
        self._version = version
        self.calibrated_sensor = {s["token"]: s for s in read_json(self._root / self._version / "calibrated_sensor.json")}
        self.category = {s["token"]: s for s in read_json(self._root / self._version / "category.json")}
        self.ego_pose = {s["token"]: s for s in read_json(self._root / self._version / "ego_pose.json")}
        self.instance = {s["token"]: s for s in read_json(self._root / self._version / "instance.json")}
        self.sample = {s["token"]: s for s in read_json(self._root / self._version / "sample.json")}
        self.sample_annotation = {s["token"]: s for s in read_json(self._root / self._version / "sample_annotation.json")}
        self.sample_data = {s["token"]: s for s in read_json(self._root / self._version / "sample_data.json")}
        self.scene = {s["token"]: s for s in read_json(self._root / self._version / "scene.json")}
        self.sensor = {s["token"]: s for s in read_json(self._root / self._version / "sensor.json")}
        self.attribute = {s["token"]: s for s in read_json(self._root / self._version / "attribute.json")}
        self.splits = read_json(self._root / self._version / "splits.json")
        self._create_relationships()

    def _add_data_to_sample(self) -> None:
        sample_data_from_sample: dict[str, dict[str, str]] = defaultdict(dict)

        for sample_data_record in self.sample_data.values():
            if not sample_data_record["is_key_frame"]:
                continue
            sample_data_from_sample[sample_data_record["sample_token"]][sample_data_record["channel"]] = sample_data_record["token"]

        for sample_record in self.sample.values():
            sample_record["data"] = sample_data_from_sample.get(sample_record["token"], {})

    def _add_channel_and_modality_to_sample_data(self):
        for sample_data_record in self.sample_data.values():
            calibrated_sensor = self.calibrated_sensor[sample_data_record["calibrated_sensor_token"]]
            sensor = self.sensor[calibrated_sensor["sensor_token"]]
            sample_data_record["channel"] = sensor["channel"]
            sample_data_record["modality"] = sensor["modality"]

    def _add_annotations_to_sample(self) -> None:
        sample_annotation_from_sample: dict[str, list[str]] = defaultdict(list)

        for sample_annotation_record in self.sample_annotation.values():
            sample_annotation_from_sample[sample_annotation_record["sample_token"]].append(sample_annotation_record["token"])

        for sample_record in self.sample.values():
            sample_record["anns"] = sample_annotation_from_sample.get(sample_record["token"], [])

    def _add_category_to_annotations(self) -> None:
        for sample_annotation_record in self.sample_annotation.values():
            instance_record = self.instance[sample_annotation_record["instance_token"]]
            category_record = self.category[instance_record["category_token"]]
            sample_annotation_record["category_name"] = category_record["name"]

    def _create_relationships(self):
        self._add_channel_and_modality_to_sample_data()
        self._add_data_to_sample()
        self._add_annotations_to_sample()
        self._add_category_to_annotations()

    def get_sensor_data_path(self, sample_data):
        return Path(self._root) / str(sample_data["filename"])

    def get_box(self, sample_annotation_token: str) -> Box:
        """
        Instantiates a Box class from a sample annotation record.
        :param sample_annotation_token: Unique sample_annotation identifier.
        """
        record = self.sample_annotation[sample_annotation_token]
        return Box(record['translation'], record['size'], Quaternion(record['rotation']),
                   name=record['category_name'], token=record['token'])

    def get_boxes(self, sample_data_token: str) -> List[Box]:
        """
        Instantiates Boxes for all annotation for a particular sample_data record. If the sample_data is a
        keyframe, this returns the annotations for that sample. But if the sample_data is an intermediate
        sample_data, a linear interpolation is applied to estimate the location of the boxes at the time the
        sample_data was captured.
        :param sample_data_token: Unique sample_data identifier.
        """

        # Retrieve sensor & pose records
        sd_record = self.sample_data[sample_data_token]
        curr_sample_record = self.sample[sd_record['sample_token']]

        if curr_sample_record['prev'] == "" or sd_record['is_key_frame']:
            # If no previous annotations available, or if sample_data is keyframe just return the current ones.
            boxes = list(map(self.get_box, curr_sample_record['anns']))

        else:
            prev_sample_record = self.sample[curr_sample_record['prev']]

            curr_ann_recs = [self.sample_annotation[token] for token in curr_sample_record['anns']]
            prev_ann_recs = [self.sample_annotation[token] for token in prev_sample_record['anns']]

            # Maps instance tokens to prev_ann records
            prev_inst_map = {entry['instance_token']: entry for entry in prev_ann_recs}

            t0 = prev_sample_record['timestamp']
            t1 = curr_sample_record['timestamp']
            t = sd_record['timestamp']

            # There are rare situations where the timestamps in the DB are off so ensure that t0 < t < t1.
            t = max(t0, min(t1, t))

            boxes = []
            for curr_ann_rec in curr_ann_recs:

                if curr_ann_rec['instance_token'] in prev_inst_map:
                    # If the annotated instance existed in the previous frame, interpolate center & orientation.
                    prev_ann_rec = prev_inst_map[curr_ann_rec['instance_token']]

                    # Interpolate center.
                    center = [np.interp(t, [t0, t1], [c0, c1]) for c0, c1 in zip(prev_ann_rec['translation'],
                                                                                 curr_ann_rec['translation'])]

                    # Interpolate orientation.
                    rotation = Quaternion.slerp(q0=Quaternion(prev_ann_rec['rotation']),
                                                q1=Quaternion(curr_ann_rec['rotation']),
                                                amount=(t - t0) / (t1 - t0))

                    box = Box(center, curr_ann_rec['size'], rotation, name=curr_ann_rec['category_name'],
                              token=curr_ann_rec['token'])
                else:
                    # If not, simply grab the current annotation.
                    box = self.get_box(curr_ann_rec['token'])

                boxes.append(box)
        return boxes

def read_json(file_path):
    """Reads json from file

    :param file_path: source file path
    :return: loaded JSON data
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def boxes_lidar_to_darts(det_info):
    boxes3d = det_info['boxes_lidar']
    scores = det_info['score']
    labels = det_info['pred_labels']

    box_list = []
    for k in range(boxes3d.shape[0]):
        quat = Quaternion(axis=[0, 0, 1], radians=boxes3d[k, 6])
        box = Box(
            boxes3d[k, :3],
            boxes3d[k, [4, 3, 5]],  # wlh
            quat, label=labels[k], score=scores[k]
        )
        box_list.append(box)
    return box_list

def lidar_darts_box_to_global(darts, boxes, sample_token):
    s_record = darts.sample[sample_token]
    sample_data_token = s_record['data']['LIDAR_TOP']

    sd_record = darts.sample_data[sample_data_token]
    cs_record = darts.calibrated_sensor[sd_record['calibrated_sensor_token']]
    pose_record = darts.ego_pose[sd_record['ego_pose_token']]

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


def transform_det_annos_to_darts_annos(det_annos, darts):
    darts_annos_by_sample = {}
    darts_annos = {"sequences": {}}

    for det in det_annos:
        annos = []
        sample_token=det['metadata']['token']
        darts_annos_by_sample[sample_token] = []
        box_list = boxes_lidar_to_darts(det)
        box_list = lidar_darts_box_to_global(
            darts=darts, boxes=box_list, sample_token=sample_token
        )

        for k, box in enumerate(box_list):
            name = det['name'][k]
            darts_anno = {
                'center': box.center.tolist(),
                'size': box.wlh.tolist(),
                'orientation': box.orientation.elements.tolist(),
                'name': name,
                'score': box.score,
            }
            annos.append(darts_anno)

        darts_annos_by_sample[sample_token] = annos
    
    val_split = darts.splits["val"]
    for scene in darts.scene.values():
        if scene["name"] not in val_split:
            continue
        sample = darts.sample[scene["first_sample_token"]]
        frames = []
        while sample["next"] != "":
            frames.append({"sample_token": sample["token"], "boxes": darts_annos_by_sample[sample["token"]]})
            sample = darts.sample[sample["next"]]
        frames.append({"sample_token": sample["token"], "boxes": darts_annos_by_sample[sample["token"]]})
        darts_annos["sequences"][scene["token"]] = frames

    return darts_annos

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

def obtain_sensor2top(
    darts, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, sensor_type="lidar"
):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = darts.sample_data[sensor_token]
    cs_record = darts.calibrated_sensor[sd_rec["calibrated_sensor_token"]]
    pose_record = darts.ego_pose[sd_rec["ego_pose_token"]]
    data_path = str(darts.get_sensor_data_path(sd_rec))
    sweep = {
        "data_path": data_path,
        "type": sensor_type,
        "sample_data_token": sd_rec["token"],
        "sensor2ego_translation": cs_record["translation"],
        "sensor2ego_rotation": cs_record["rotation"],
        "ego2global_translation": pose_record["translation"],
        "ego2global_rotation": pose_record["rotation"],
        "timestamp": sd_rec["timestamp"],
    }
    l2e_r_s = sweep["sensor2ego_rotation"]
    l2e_t_s = sweep["sensor2ego_translation"]
    e2g_r_s = sweep["ego2global_rotation"]
    e2g_t_s = sweep["ego2global_translation"]

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    ).squeeze(0)
    sweep["sensor2lidar_rotation"] = R.T  # points @ R.T + T
    sweep["sensor2lidar_translation"] = T
    return sweep

def get_sample_data(darts, sample_data_token, selected_anntokens=None):
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
    sd_record = darts.sample_data[sample_data_token]
    cs_record = darts.calibrated_sensor[sd_record['calibrated_sensor_token']]
    sensor_record = darts.sensor[cs_record['sensor_token']]
    pose_record = darts.ego_pose[sd_record['ego_pose_token']]

    data_path = darts.get_sensor_data_path(sd_record)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
    else:
        cam_intrinsic = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    if selected_anntokens is not None:
        boxes = list(map(darts.get_box, selected_anntokens))
    else:
        boxes = darts.get_boxes(sample_data_token)

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

def fill_trainval_infos(data_path, darts, train_scenes, val_scenes, with_cam=False):
    train_nusc_infos = []
    val_nusc_infos = []
    progress_bar = tqdm.tqdm(total=len(darts.sample), desc='create_info', dynamic_ncols=True)

    ref_chan = 'LIDAR_TOP'  # The radar channel from which we track back n sweeps to aggregate the point cloud.

    for sample in darts.sample.values():
        if sample["scene_token"] not in val_scenes and sample["scene_token"] not in train_scenes:
            continue
        progress_bar.update()

        ref_sd_token = sample['data'][ref_chan]
        ref_sd_rec = darts.sample_data[ref_sd_token]
        ref_cs_rec = darts.calibrated_sensor[ref_sd_rec['calibrated_sensor_token']]
        ref_pose_rec = darts.ego_pose[ref_sd_rec['ego_pose_token']]
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        ref_lidar_path, ref_boxes, _ = get_sample_data(darts, ref_sd_token)

        ref_cam_front_token = sample['data']['CAM_FRONT']
        ref_cam_path, _, ref_cam_intrinsic = get_sample_data(darts, ref_cam_front_token)

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
            'ref_from_car': ref_from_car,
            'car_from_global': car_from_global,
            'timestamp': ref_time,
        }
        if with_cam:
            info['cams'] = dict()
            l2e_r = ref_cs_rec["rotation"]
            l2e_t = ref_cs_rec["translation"],
            e2g_r = ref_pose_rec["rotation"]
            e2g_t = ref_pose_rec["translation"]
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
            e2g_r_mat = Quaternion(e2g_r).rotation_matrix

            # obtain 7 image's information per frame
            camera_types = [
                "CAM_FRONT",
                "CAM_FRONT_TELE",
                "CAM_FRONT_RIGHT",
                "CAM_FRONT_LEFT",
                "CAM_BACK",
                "CAM_BACK_LEFT",
                "CAM_BACK_RIGHT",
            ]
            for cam in camera_types:
                cam_token = sample["data"][cam]
                _, _, camera_intrinsics = get_sample_data(darts, cam_token)
                cam_info = obtain_sensor2top(
                    darts, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam
                )
                cam_info['data_path'] = Path(cam_info['data_path']).relative_to(data_path).__str__()
                cam_info.update(camera_intrinsics=camera_intrinsics)
                info["cams"].update({cam: cam_info})

        annotations = [darts.sample_annotation[token] for token in sample['anns']]

        num_lidar_pts = np.array([anno['num_lidar_pts'] for anno in annotations])
        locs = np.array([b.center for b in ref_boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in ref_boxes]).reshape(-1, 3)[:, [1, 0, 2]]  # wlh == > dxdydz (lwh)
        rots = np.array([quaternion_yaw(b.orientation) for b in ref_boxes]).reshape(-1, 1)
        names = np.array([b.name for b in ref_boxes])
        tokens = np.array([b.token for b in ref_boxes])
        gt_boxes = np.concatenate([locs, dims, rots], axis=1)

        assert len(annotations) == len(gt_boxes)

        info['gt_boxes'] = gt_boxes
        info['gt_names'] = names
        info['gt_boxes_token'] = tokens
        info['num_lidar_pts'] = num_lidar_pts

        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    progress_bar.close()
    return train_nusc_infos, val_nusc_infos
