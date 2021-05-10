import os
import numpy as np
import math

from pcdet.datasets.neolix.neolix_object_eval_python.rotate_iou import rotate_iou_gpu_eval

ACCURACY_DICT = {}
MIN_OVERLAP = {'Vehicle': 0.5, 'Pedestrian': 0.25, 'Cyclist': 0.25, 'Unknown': 0.25, 'Large_vehicle': 0.5}

def get_boxes(gt_file, dt_file):
    """
    Get boxes from label file(.txt)
    Args:
        gt_file:
        dt_file:

    Returns:

    """
    gt_boxes = []
    dt_boxes = []
    gt_names = []
    dt_names = []
    with open(gt_file, 'r') as f_g:
        gt_lines = f_g.readlines()
        for gt_line in gt_lines:
            line_parts = gt_line.strip().split(' ')
            gt_boxes.append(list(map(float, [line_parts[11], line_parts[12], line_parts[10], line_parts[9], -np.pi/2-float(line_parts[-1])])))
            gt_names.append(line_parts[0])
    with open(dt_file, 'r') as f_d:
        dt_lines = f_d.readlines()
        for dt_line in dt_lines:
            line_parts = dt_line.strip().split(' ')
            dt_boxes.append(list(map(float, [line_parts[11], line_parts[12], line_parts[10], line_parts[9], line_parts[-2]])))
            dt_names.append(line_parts[0])
    return np.array(gt_boxes), np.array(dt_boxes), np.array(gt_names), np.array(dt_names)


def bev_box_overlap(boxes, qboxes, criterion=-1):
    """
    Calculate rotated 2D iou.
    Args:
        boxes:
        qboxes:
        criterion:

    Returns:

    """
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou


def format_rotate(theta):
    """

    Args:
        theta:

    Returns:

    """
    tmp_theta = theta % np.pi
    if tmp_theta < 0:
        return tmp_theta + np.pi
    else:
        return tmp_theta


def distance(p1, p2, axis_mode=0):
    """

    Args:
        p1: point1 [x, y]
        p2: point2 [x, y]
        axis_mode: {0, 1, 2} 0: horizontal distance,  1:  Vertical  distance,  2: Euler distance

    Returns:

    """
    if axis_mode == 0:
        return abs(p1[0] - p2[0])
    elif axis_mode == 1:
        return abs(p1[1] - p2[1])
    elif axis_mode == 2:
        return math.sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))
    else:
        print('error distance mode')


def cal_accuracy(gt_boxes, dt_boxes, gt_names, dt_names, overlaps):
    """
    Calculate position accuracy.
    Args:
        gt_boxes:
        dt_boxes:
        gt_names:   ['Vehicle', 'Pedestrian', 'Cyclist', 'Unknown', 'Large_vehicle']
        dt_names:
        overlaps:

    Returns:

    """
    def rotate_point(x1, y1, x0, y0, t):
        # x = (x1 - x2) cosθ - (y1 - y2) sinθ + x2
        # y = (y1 - y2) cosθ + (x1 - x2) sinθ + y2
        x2 = (x1 - x0) * math.cos(t) - (y1 - y0) * math.sin(t) + x0
        y2 = (y1 - y0) * math.cos(t) + (x1 - x0) * math.sin(t) + y0
        return [x2, y2]

    gt_num = gt_boxes.shape[0]
    overlaps = overlaps.tolist()
    for i in range(gt_num):
        gt_box = gt_boxes[i]
        gt_name = gt_names[i]
        optional_overlap = max(overlaps[i])
        dt_box = dt_boxes[overlaps[i].index(optional_overlap)]
        dt_name = dt_names[overlaps[i].index(optional_overlap)]
        if (optional_overlap >= MIN_OVERLAP[gt_name]) and ((gt_name == dt_name) or ((gt_name in ['Vehicle', 'Large_vehicle']) and (dt_name in ['Vehicle', 'Large_vehicle']))):
            gt_box[-1] = format_rotate(float(gt_box[-1]))
            dt_box[-1] = format_rotate(float(dt_box[-1]))
            x, y, l, w, t1 = gt_box
            p1 = [[x-l/2, y-w/2], [x-l/2, y+w/2], [x+l/2, y-w/2], [ x+l/2, y+w/2]]
            p1 = [rotate_point(x1, y1, x, y, t1) for x1, y1 in p1]
            x, y, l, w, t2 = dt_box
            p2 = [[x-l/2, y-w/2], [x-l/2, y+w/2], [x+l/2, y-w/2], [ x+l/2, y+w/2]]
            if abs(t1 - t2) > np.pi/2:
                t2 = t1 - 2 * np.pi
            p2 = [rotate_point(x1, y1, x, y, t2) for x1, y1 in p2]

            distance_ls = []
            global ACCURACY_DICT
            for i in range(len(p1)):
                distance_ls.append(distance(p1[i], p2[i], 1))
            if gt_name in ACCURACY_DICT.keys():
                ACCURACY_DICT[gt_name] = ACCURACY_DICT[gt_name] + [[max(distance_ls) , gt_box[0], gt_box[1]]]
            else:
                ACCURACY_DICT[gt_name] = [[max(distance_ls), gt_box[0], gt_box[1]]]



def main():
    gt_path = '/nfs/neolix_data1/neolix_dataset/develop_dataset/lidar_object_detection/ID_1022/training/val_label/'
    dt_path = '/home/liuwanqiang/OpenPCDet-master/OpenPCDet-master/tools/val_label/'
    dt_ls = sorted(os.listdir(dt_path))
    dt_range_ls = None
    dt_range_ls = [[-10, 10, 0, 10], [-10, 10, 10, 30], [-10, 10, 30, 50], [-10, 10, 50, 70]]
    for dt in dt_ls:
        dt_file = dt_path + dt
        gt_file = gt_path + dt
        gt_boxes, dt_boxes, gt_names, dt_names = get_boxes(gt_file, dt_file)
        overlap = bev_box_overlap(gt_boxes, dt_boxes).astype(np.float64)
        cal_accuracy(gt_boxes, dt_boxes, gt_names, dt_names, overlap)
    if dt_range_ls is None:
        for cls in ACCURACY_DICT.keys():
            print(ACCURACY_DICT[cls])
            print("%d %s boxes, mean position accuracy %f" % (len(ACCURACY_DICT[cls]), cls, np.array(ACCURACY_DICT[cls])[:, 0].mean()))
    else:
        for dt_range in dt_range_ls:
            print("The range of detection is %d < x < %d, %d < abs(y) < %d" % (
            dt_range[0], dt_range[1], dt_range[2], dt_range[3]))
            for cls in ACCURACY_DICT.keys():
                optional_box = []
                for box in ACCURACY_DICT[cls]:
                    if (dt_range[0] < box[1] < dt_range[1]) and (abs(dt_range[2]) < box[2] < abs(dt_range[3])):
                        optional_box.append(box)
                print("%d %s boxes, mean position accuracy %f" % (len(optional_box), cls, np.array(optional_box)[:, 0].mean()))


if __name__ == '__main__':
    main()