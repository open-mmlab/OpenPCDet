import sys, os, copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image

# import mayavi.mlab as mlab
from vis_utils import calibration_kitti, object3d_kitti, visualizer

'''
 Values  Name        Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
'''

# 3d visualization
DATA_PATH = '/home/qxl/DATACENTER/kitti/training'

image_set = os.listdir(os.path.join(DATA_PATH, 'image_2'))


# print (len(image_set))

def get_image(idx):
    img_file = os.path.join(DATA_PATH, 'image_2', idx + '.png')
    return cv2.imread(img_file)
    # return  Image.open(img_file).convert('RGB')


def get_calib(idx):
    calib_file = os.path.join(DATA_PATH, 'calib', idx + '.txt')
    return calibration_kitti.Calibration(calib_file)


def get_label(idx):
    label_file = os.path.join(DATA_PATH, 'label_2', idx + '.txt')
    return object3d_kitti.get_objects_from_label(label_file)


def get_lidar(idx):
    lidar_file = os.path.join(DATA_PATH, 'velodyne', idx + '.bin')
    return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)


def get_fov_flag(pts_rect, img_shape, calib):
    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

    return pts_valid_flag


# Front-view Image
def Image_Vis_2d():
    for name in image_set:
        sample_idx = os.path.splitext(name)[0]
        img = get_image(sample_idx)

        calib = get_calib(sample_idx)
        anno = get_label(sample_idx)

        # print (len(anno))
        print('The anno of %s:' % (name))
        for a in anno:
            print('    %s' % (a.to_str()))

            loc = a.loc
            loc = calib.rect_to_img(loc[None])[0]
            cv2.circle(img, (loc[0, 0], loc[0, 1]), 2, color=(255, 0, 0), thickness=2)
            cv2.imshow('1', img)
            cv2.waitKey()

            box_2d = a.box2d
            img = visualizer.draw_bbox_2d(img, a.cls_id, a.cls_type, box_2d)
            cv2.imshow('1', img)
            cv2.waitKey()

            corners3d = a.generate_corners3d()
            pts_img, depth_img = calib.rect_to_img(corners3d)
            img = visualizer.draw_bbox_3d_img(img, pts_img)
            cv2.imshow('1', img)
            cv2.waitKey()


# BEV
def Lidar_Vis_3d():
    for name in image_set:
        sample_idx = os.path.splitext(name)[0]
        points = get_lidar(sample_idx)
        print(sample_idx)
        if sample_idx != '004409':
            continue

        calib = get_calib(sample_idx)
        anno = get_label(sample_idx)

        # plt.figure(dpi=120)
        pc_range = [0, -40, -3, 70.4, 40, 1]
        print(points.shape)
        mask = (points[:, 0] >= pc_range[0]) & (points[:, 0] <= pc_range[3]) \
               & (points[:, 1] >= pc_range[1]) & (points[:, 1] <= pc_range[4])
        points = points[mask]

        # N = points.shape[0]
        # print (N)
        # sample_n = np.random.choice(np.arange(N), size=64, replace=64>N)
        # points = points[sample_n]

        rgba_colors = np.zeros((points.shape[0], 4))
        rgba_colors[:, 2] = 1
        rgba_colors[:, 3] = points[:, 3]
        plt.scatter(points[:, 0], points[:, 1], s=0.5, color=rgba_colors[:, :3])

        for a in anno:
            if a.cls_type == 'DontCare':
                continue
            print('    %s %s: %.3f' % (a.to_str(), 'head', -(np.pi / 2 + a.ry)))

            # a.ry = np.pi / 2
            corners3d = a.generate_corners3d()
            pts_lidar = calib.rect_to_lidar(corners3d)
            for i in range(5):
                if i == 3:
                    plt.plot((pts_lidar[i, 0], pts_lidar[0, 0]), (pts_lidar[i, 1], pts_lidar[0, 1]), color='red',
                             linewidth=1)
                elif i == 4:
                    x1 = ((pts_lidar[3, 0] - pts_lidar[0, 0]) / 3 + pts_lidar[0, 0] + (
                                pts_lidar[2, 0] - pts_lidar[1, 0]) / 3 + pts_lidar[1, 0]) / 2
                    x2 = ((pts_lidar[0, 0] - pts_lidar[3, 0]) / 4 + pts_lidar[0, 0] + (
                                pts_lidar[1, 0] - pts_lidar[2, 0]) / 4 + pts_lidar[1, 0]) / 2
                    y1 = ((pts_lidar[3, 1] - pts_lidar[0, 1]) / 3 + pts_lidar[0, 1] + (
                                pts_lidar[2, 1] - pts_lidar[1, 1]) / 3 + pts_lidar[1, 1]) / 2
                    y2 = ((pts_lidar[0, 1] - pts_lidar[3, 1]) / 4 + pts_lidar[0, 1] + (
                                pts_lidar[1, 1] - pts_lidar[2, 1]) / 4 + pts_lidar[1, 1]) / 2
                    plt.plot((x1, x2), (y1, y2), color='yellowgreen', linewidth=2)
                else:
                    plt.plot((pts_lidar[i, 0], pts_lidar[i + 1, 0]), (pts_lidar[i, 1], pts_lidar[i + 1, 1]),
                             color='red', linewidth=1)

        plt.show()


if __name__ == '__main__':
    # Image_Vis_2d()
    Lidar_Vis_3d()
