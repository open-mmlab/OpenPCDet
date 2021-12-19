"""
modified from lyft toolkit https://github.com/lyft/nuscenes-devkit.git
"""

"""
mAP 3D calculation for the data in nuScenes format.


The intput files expected to have the format:

Expected fields:


gt = [{
    'sample_token': '0f0e3ce89d2324d8b45aa55a7b4f8207fbb039a550991a5149214f98cec136ac',
    'translation': [974.2811881299899, 1714.6815014457964, -23.689857123368846],
    'size': [1.796, 4.488, 1.664],
    'rotation': [0.14882026466054782, 0, 0, 0.9888642620837121],
    'name': 'car'
}]

prediction_result = {
    'sample_token': '0f0e3ce89d2324d8b45aa55a7b4f8207fbb039a550991a5149214f98cec136ac',
    'translation': [971.8343488872263, 1713.6816097857359, -25.82534357061308],
    'size': [2.519726579986132, 7.810161372666739, 3.483438286096803],
    'rotation': [0.10913582721095375, 0.04099572636992043, 0.01927712319721745, 1.029328402625659],
    'name': 'car',
    'score': 0.3077029437237213
}


input arguments:

--pred_file:  file with predictions
--gt_file: ground truth file
--iou_threshold: IOU threshold


In general we would be interested in average of mAP at thresholds [0.5, 0.55, 0.6, 0.65,...0.95], similar to the
standard COCO => one needs to run this file N times for every IOU threshold independently.

"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion
from shapely.geometry import Polygon


class Box3D:
    """Data class used during detection evaluation. Can be a prediction or ground truth."""

    def __init__(self, **kwargs):
        sample_token = kwargs["sample_token"]
        translation = kwargs["translation"]
        size = kwargs["size"]
        rotation = kwargs["rotation"]
        name = kwargs["name"]
        score = kwargs.get("score", -1)

        if not isinstance(sample_token, str):
            raise TypeError("Sample_token must be a string!")

        if not len(translation) == 3:
            raise ValueError("Translation must have 3 elements!")

        if np.any(np.isnan(translation)):
            raise ValueError("Translation may not be NaN!")

        if not len(size) == 3:
            raise ValueError("Size must have 3 elements!")

        if np.any(np.isnan(size)):
            raise ValueError("Size may not be NaN!")

        if not len(rotation) == 4:
            raise ValueError("Rotation must have 4 elements!")

        if np.any(np.isnan(rotation)):
            raise ValueError("Rotation may not be NaN!")

        if name is None:
            raise ValueError("Name cannot be empty!")

        # Assign.
        self.sample_token = sample_token
        self.translation = translation
        self.size = size
        self.volume = np.prod(self.size)
        self.score = score

        assert np.all([x > 0 for x in size])
        self.rotation = rotation
        self.name = name
        self.quaternion = Quaternion(self.rotation)

        self.width, self.length, self.height = size

        self.center_x, self.center_y, self.center_z = self.translation

        self.min_z = self.center_z - self.height / 2
        self.max_z = self.center_z + self.height / 2

        self.ground_bbox_coords = None
        self.ground_bbox_coords = self.get_ground_bbox_coords()

    @staticmethod
    def check_orthogonal(a, b, c):
        """Check that vector (b - a) is orthogonal to the vector (c - a)."""
        return np.isclose((b[0] - a[0]) * (c[0] - a[0]) + (b[1] - a[1]) * (c[1] - a[1]), 0)

    def get_ground_bbox_coords(self):
        if self.ground_bbox_coords is not None:
            return self.ground_bbox_coords
        return self.calculate_ground_bbox_coords()

    def calculate_ground_bbox_coords(self):
        """We assume that the 3D box has lower plane parallel to the ground.

        Returns: Polygon with 4 points describing the base.

        """
        if self.ground_bbox_coords is not None:
            return self.ground_bbox_coords

        rotation_matrix = self.quaternion.rotation_matrix

        cos_angle = rotation_matrix[0, 0]
        sin_angle = rotation_matrix[1, 0]

        point_0_x = self.center_x + self.length / 2 * cos_angle + self.width / 2 * sin_angle
        point_0_y = self.center_y + self.length / 2 * sin_angle - self.width / 2 * cos_angle

        point_1_x = self.center_x + self.length / 2 * cos_angle - self.width / 2 * sin_angle
        point_1_y = self.center_y + self.length / 2 * sin_angle + self.width / 2 * cos_angle

        point_2_x = self.center_x - self.length / 2 * cos_angle - self.width / 2 * sin_angle
        point_2_y = self.center_y - self.length / 2 * sin_angle + self.width / 2 * cos_angle

        point_3_x = self.center_x - self.length / 2 * cos_angle + self.width / 2 * sin_angle
        point_3_y = self.center_y - self.length / 2 * sin_angle - self.width / 2 * cos_angle

        point_0 = point_0_x, point_0_y
        point_1 = point_1_x, point_1_y
        point_2 = point_2_x, point_2_y
        point_3 = point_3_x, point_3_y

        assert self.check_orthogonal(point_0, point_1, point_3)
        assert self.check_orthogonal(point_1, point_0, point_2)
        assert self.check_orthogonal(point_2, point_1, point_3)
        assert self.check_orthogonal(point_3, point_0, point_2)

        self.ground_bbox_coords = Polygon(
            [
                (point_0_x, point_0_y),
                (point_1_x, point_1_y),
                (point_2_x, point_2_y),
                (point_3_x, point_3_y),
                (point_0_x, point_0_y),
            ]
        )

        return self.ground_bbox_coords

    def get_height_intersection(self, other):
        min_z = max(other.min_z, self.min_z)
        max_z = min(other.max_z, self.max_z)

        return max(0, max_z - min_z)

    def get_area_intersection(self, other) -> float:
        result = self.ground_bbox_coords.intersection(other.ground_bbox_coords).area

        assert result <= self.width * self.length

        return result

    def get_intersection(self, other) -> float:
        height_intersection = self.get_height_intersection(other)

        area_intersection = self.ground_bbox_coords.intersection(other.ground_bbox_coords).area

        return height_intersection * area_intersection

    def get_iou(self, other):
        intersection = self.get_intersection(other)
        union = self.volume + other.volume - intersection

        iou = np.clip(intersection / union, 0, 1)

        return iou

    def __repr__(self):
        return str(self.serialize())

    def serialize(self) -> dict:
        """Returns: Serialized instance as dict."""

        return {
            "sample_token": self.sample_token,
            "translation": self.translation,
            "size": self.size,
            "rotation": self.rotation,
            "name": self.name,
            "volume": self.volume,
            "score": self.score,
        }


def group_by_key(detections, key):
    groups = defaultdict(list)
    for detection in detections:
        groups[detection[key]].append(detection)
    return groups


def wrap_in_box(input):
    result = {}
    for key, value in input.items():
        result[key] = [Box3D(**x) for x in value]

    return result


def get_envelope(precisions):
    """Compute the precision envelope.

    Args:
      precisions:

    Returns:

    """
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
    return precisions


def get_ap(recalls, precisions):
    """Calculate average precision.

    Args:
      recalls:
      precisions: Returns (float): average precision.

    Returns:

    """
    # correct AP calculation
    # first append sentinel values at the end
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    precisions = get_envelope(precisions)

    # to calculate area under PR curve, look for points where X axis (recall) changes value
    i = np.where(recalls[1:] != recalls[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
    return ap


def get_ious(gt_boxes, predicted_box):
    return [predicted_box.get_iou(x) for x in gt_boxes]


def recall_precision(gt, predictions, iou_threshold_list):
    num_gts = len(gt)

    if num_gts == 0:
        return -1, -1, -1

    image_gts = group_by_key(gt, "sample_token")
    image_gts = wrap_in_box(image_gts)

    sample_gt_checked = {sample_token: np.zeros((len(boxes), len(iou_threshold_list))) for sample_token, boxes in image_gts.items()}

    predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)

    # go down dets and mark TPs and FPs
    num_predictions = len(predictions)
    tp = np.zeros((num_predictions, len(iou_threshold_list)))
    fp = np.zeros((num_predictions, len(iou_threshold_list)))

    for prediction_index, prediction in enumerate(predictions):
        predicted_box = Box3D(**prediction)

        sample_token = prediction["sample_token"]

        max_overlap = -np.inf
        jmax = -1

        try:
            gt_boxes = image_gts[sample_token]  # gt_boxes per sample
            gt_checked = sample_gt_checked[sample_token]  # gt flags per sample
        except KeyError:
            gt_boxes = []
            gt_checked = None

        if len(gt_boxes) > 0:
            overlaps = get_ious(gt_boxes, predicted_box)

            max_overlap = np.max(overlaps)

            jmax = np.argmax(overlaps)

        for i, iou_threshold in enumerate(iou_threshold_list):
            if max_overlap > iou_threshold:
                if gt_checked[jmax, i] == 0:
                    tp[prediction_index, i] = 1.0
                    gt_checked[jmax, i] = 1
                else:
                    fp[prediction_index, i] = 1.0
            else:
                fp[prediction_index, i] = 1.0

    # compute precision recall
    fp = np.cumsum(fp, axis=0)
    tp = np.cumsum(tp, axis=0)

    recalls = tp / float(num_gts)

    assert np.all(0 <= recalls) & np.all(recalls <= 1)

    # avoid divide by zero in case the first detection matches a difficult ground truth
    precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    assert np.all(0 <= precisions) & np.all(precisions <= 1)

    ap_list = []
    for i in range(len(iou_threshold_list)):
        recall = recalls[:, i]
        precision = precisions[:, i]
        ap = get_ap(recall, precision)
        ap_list.append(ap)

    return recalls, precisions, ap_list


def get_average_precisions(gt: list, predictions: list, class_names: list, iou_thresholds: list) -> np.array:
    """Returns an array with an average precision per class.


    Args:
        gt: list of dictionaries in the format described below.
        predictions: list of dictionaries in the format described below.
        class_names: list of the class names.
        iou_threshold: list of IOU thresholds used to calculate TP / FN

    Returns an array with an average precision per class.


    Ground truth and predictions should have schema:

    gt = [{
    'sample_token': '0f0e3ce89d2324d8b45aa55a7b4f8207fbb039a550991a5149214f98cec136ac',
    'translation': [974.2811881299899, 1714.6815014457964, -23.689857123368846],
    'size': [1.796, 4.488, 1.664],
    'rotation': [0.14882026466054782, 0, 0, 0.9888642620837121],
    'name': 'car'
    }]

    predictions = [{
        'sample_token': '0f0e3ce89d2324d8b45aa55a7b4f8207fbb039a550991a5149214f98cec136ac',
        'translation': [971.8343488872263, 1713.6816097857359, -25.82534357061308],
        'size': [2.519726579986132, 7.810161372666739, 3.483438286096803],
        'rotation': [0.10913582721095375, 0.04099572636992043, 0.01927712319721745, 1.029328402625659],
        'name': 'car',
        'score': 0.3077029437237213
    }]

    """
    assert all([0 <= iou_th <= 1 for iou_th in iou_thresholds])

    gt_by_class_name = group_by_key(gt, "name")
    pred_by_class_name = group_by_key(predictions, "name")

    average_precisions = np.zeros(len(class_names))

    for class_id, class_name in enumerate(class_names):
        if class_name in pred_by_class_name:
            recalls, precisions, ap_list = recall_precision(
                gt_by_class_name[class_name], pred_by_class_name[class_name], iou_thresholds
            )
            aps = np.mean(ap_list)
            average_precisions[class_id] = aps

    return average_precisions


def get_class_names(gt: dict) -> list:
    """Get sorted list of class names.

    Args:
        gt:

    Returns: Sorted list of class names.

    """
    return sorted(list(set([x["name"] for x in gt])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-p", "--pred_file", type=str, help="Path to the predictions file.", required=True)
    arg("-g", "--gt_file", type=str, help="Path to the ground truth file.", required=True)
    arg("-t", "--iou_threshold", type=float, help="iou threshold", default=0.5)

    args = parser.parse_args()

    gt_path = Path(args.gt_file)
    pred_path = Path(args.pred_file)

    with open(args.pred_file) as f:
        predictions = json.load(f)

    with open(args.gt_file) as f:
        gt = json.load(f)

    class_names = get_class_names(gt)
    print("Class_names = ", class_names)

    average_precisions = get_average_precisions(gt, predictions, class_names, args.iou_threshold)

    mAP = np.mean(average_precisions)
    print("Average per class mean average precision = ", mAP)

    for class_id in sorted(list(zip(class_names, average_precisions.flatten().tolist()))):
        print(class_id)
