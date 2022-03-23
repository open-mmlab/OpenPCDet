import numpy as np
import cv2
import torch


def convert_voxel_to_bev_map(voxel_coords, voxel_size, area_scope=[[-50, 50], [-50, 50], [-5, 5]]):
    """
    Args: 
        voxel_coords: non-empty voxel's coord. (N, 3) [x,y,z]
        voxel_size: list 
        area_scope: list
    Return:
        voxel_map: 
    """
    voxel_coords = np.array(voxel_coords).astype(np.int32)
    voxel_size = np.array(voxel_size)
    area_scope = np.array(area_scope)

    voxelized_shape = ((area_scope[:,1] - area_scope[:,0])/voxel_size).astype(np.int32)
    
    voxel_map = np.zeros(voxelized_shape, dtype=np.float32)
    voxel_map[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = 1.0
    voxel_map = np.flip(np.flip(voxel_map, axis=0), axis=1)
    return voxel_map

def boxes3d_to_corners3d_lidar(boxes3d):
    """
    :param boxes3d: (N, 7) [x, y, z, w, l, h, ry] in LiDAR coords
    :param rotate:
    :return: corners3d: (N, 8, 3)
    """
    boxes_num = boxes3d.shape[0]
    w, l, h = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners = np.array([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dtype=np.float32).T
    y_corners = np.array([-l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2.], dtype=np.float32).T
    z_corners = np.zeros((boxes_num, 8), dtype=np.float32)
    z_corners[:, 4:8] = h.reshape(boxes_num, 1).repeat(4, axis=1)  # (N, 8)

    ry = -boxes3d[:, 6]
    zeros, ones = np.zeros(ry.size, dtype=np.float32), np.ones(ry.size, dtype=np.float32)
    rot_list = np.array([[np.cos(ry), -np.sin(ry), zeros],
                         [np.sin(ry), np.cos(ry), zeros],
                         [zeros, zeros, ones]])  # (3, 3, N)
    R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

    temp_corners = np.concatenate((x_corners.reshape(-1, 8, 1), y_corners.reshape(-1, 8, 1),
                                   z_corners.reshape(-1, 8, 1)), axis=2)  # (N, 8, 3)
    rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
    x_corners, y_corners, z_corners = rotated_corners[:, :, 0], rotated_corners[:, :, 1], rotated_corners[:, :, 2]

    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2)

    return corners.astype(np.float32)

def corners3d_to_bev_corners(corners3d, voxel_size=(0.1, 0.1, 0.2), area_scope=[[-50, 50], [-50, 50], [-5, 5]]):
    """
    :param corners3d: (N, 8, 3)
    :return:
        bev_corners: (N, 4, 2)
    """
    voxel_size = np.array(voxel_size)
    area_scope = np.array(area_scope)

    voxel_idxs = np.floor(corners3d[:, :, 0:3] / voxel_size).astype(np.int32)

    min_voxel_coords = np.floor(area_scope[:, 0] / voxel_size).astype(np.int32)
    max_voxel_coords = np.ceil(area_scope[:, 1] / voxel_size - 1).astype(np.int32)
    voxelized_shape = ((max_voxel_coords - min_voxel_coords) + 1).astype(np.int32)

    # Check the points are bounded by the image scope
    # assert (min_voxel_coords <= np.amin(voxel_idxs, axis=0)).all(), 'Shape: %s' % (str(voxel_idxs.shape))
    # assert (max_voxel_coords >= np.amax(voxel_idxs, axis=0)).all(), 'Shape: %s' % (str(voxel_idxs.shape))
    voxel_idxs = voxel_idxs - min_voxel_coords
    voxel_idxs = voxel_idxs[:, 0:4, 0:2]
    x_idxs, y_idxs = voxel_idxs[:, :, 0].copy(), voxel_idxs[:, :, 1].copy()
    voxel_idxs[:, :, 0] = voxelized_shape[1] - y_idxs
    voxel_idxs[:, :, 1] = voxelized_shape[0] - x_idxs
    return voxel_idxs

def corners3d_to_bev_corners_v2(corners3d, voxel_size=(0.1, 0.1, 0.2), area_scope=[[-50, 50], [-50, 50], [-5, 5]]):
    """
    :param corners3d: (N, 8, 3)
    :return:
        bev_corners: (N, 4, 2)
    """
    voxel_size = np.array(voxel_size)
    area_scope = np.array(area_scope)

    voxel_idxs = np.floor(corners3d[:, :, 0:3] / voxel_size).astype(np.int32)

    min_voxel_coords = np.floor(area_scope[:, 0] / voxel_size).astype(np.int32)
    max_voxel_coords = np.ceil(area_scope[:, 1] / voxel_size - 1).astype(np.int32)
    voxelized_shape = ((max_voxel_coords - min_voxel_coords) + 1).astype(np.int32)

    # Check the points are bounded by the image scope
    # assert (min_voxel_coords <= np.amin(voxel_idxs, axis=0)).all(), 'Shape: %s' % (str(voxel_idxs.shape))
    # assert (max_voxel_coords >= np.amax(voxel_idxs, axis=0)).all(), 'Shape: %s' % (str(voxel_idxs.shape))
    voxel_idxs = voxel_idxs - min_voxel_coords
    voxel_idxs = voxel_idxs[:, 0:4, 0:2]
    return voxel_idxs


def draw_bev_gt(frame_bev_path, voxel_coords, gt_boxes = None, area_scope = [[-72, 92], [-72, 72], [-5, 5]], cmap_color = False, voxel_size=(0.16, 0.16, 0.2)):
    """
    Args:
        voxel_coords: np.adarray (N, 3) [x,y,z] in camera coords
        gt_boxes: (N, 7) [x, y, z, w, l, h, ry] in LiDAR coords
    """    

    voxel_map = convert_voxel_to_bev_map(voxel_coords, voxel_size=voxel_size, area_scope=area_scope)    # (w, h, 3)
    bev_map = voxel_map.sum(axis=2)
    if cmap_color:
        # bev_map = bev_map > 0
        bev_map = bev_map / max(bev_map.max(), 1.0)
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('hot')
        bev_map = cmap(bev_map)
        bev_map = np.delete(bev_map, 3, 2)
        bev_map[:, :, [0, 1, 2]] = bev_map[:, :, [2, 1, 0]] * 255
        bev_map = bev_map.astype(np.uint8)
    else:
        bev_index = bev_map > 0
        bev_map = np.zeros([bev_map.shape[0], bev_map.shape[1], 3], dtype = np.uint8)
        bev_map[bev_index] = (228, 197, 85)
    
    # classes = ['', 'Car', 'Pedestrian', 'Cyclist', 'Truck', 'Cone', 'Unknown', 'Dontcare']
    if gt_boxes is None or gt_boxes.shape[0] == 0:
        return bev_map
    gt_boxes_corner3d = boxes3d_to_corners3d_lidar(gt_boxes)  # (N, 8, 3)
    gt_boxes_corners_bev = corners3d_to_bev_corners(gt_boxes_corner3d, voxel_size=voxel_size, area_scope=area_scope)    # (N, 4, 2)

    for k in range(gt_boxes_corners_bev.shape[0]):
        for j in range(0, 4):
            x1, y1 = gt_boxes_corners_bev[k, j, 0], gt_boxes_corners_bev[k, j, 1]
            x2, y2 = gt_boxes_corners_bev[k, (j + 1)%4, 0], gt_boxes_corners_bev[k, (j + 1) % 4, 1]
            cv2.line(bev_map, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
    bev_image = bev_map.astype(np.uint8)
    
    # hard code
    bev_image = cv2.resize(bev_image, dsize=(248, 216))
    bev_image = bev_image[::-1,:,:][:,::-1,:].transpose(1,0,2)
    cv2.imwrite(frame_bev_path, bev_image)
    return bev_image



def draw_bev_pts(frame_bev_path, voxel_coords, gt_boxes = None, area_scope = [[-72, 92], [-72, 72], [-5, 5]], cmap_color = False, voxel_size=(0.16, 0.16, 0.2)):
    """
    Args:
        voxel_coords: np.adarray (N, 3) [x,y,z] in camera coords
        gt_boxes: (N, 7) [x, y, z, w, l, h, ry] in LiDAR coords
    """    

    voxel_map = convert_voxel_to_bev_map(voxel_coords, voxel_size=voxel_size, area_scope=area_scope)    # (w, h, 3)
    bev_map = voxel_map.sum(axis=2)
    if cmap_color:
        # bev_map = bev_map > 0
        bev_map = bev_map / max(bev_map.max(), 1.0)
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('hot')
        bev_map = cmap(bev_map)
        bev_map = np.delete(bev_map, 3, 2)
        bev_map[:, :, [0, 1, 2]] = bev_map[:, :, [2, 1, 0]] * 255
        bev_map = bev_map.astype(np.uint8)
    else:
        bev_index = bev_map > 0
        bev_map = np.zeros([bev_map.shape[0], bev_map.shape[1], 3], dtype = np.uint8)
        bev_map[bev_index] = (228, 197, 85)
      
    bev_image = bev_map.astype(np.uint8)

    # hard code
    bev_image = cv2.resize(bev_image, dsize=(248, 216))
    bev_image = bev_image[::-1,:,:][:,::-1,:].transpose(1,0,2)
    cv2.imwrite(frame_bev_path, bev_image)
    return bev_image

def get_bev_gt(voxel_coords, gt_boxes = None, area_scope = [[-72, 92], [-72, 72], [-5, 5]], cmap_color = False, voxel_size=(0.16, 0.16, 0.2)):
    """
    Args:
        voxel_coords: np.adarray (N, 3) [x,y,z] in camera coords
        gt_boxes: (N, 7) [x, y, z, w, l, h, ry] in LiDAR coords
    """    
    voxel_map = convert_voxel_to_bev_map(voxel_coords, voxel_size=voxel_size, area_scope=area_scope)    # (w, h, 3)
    # voxel_coords = np.array(voxel_coords).astype(np.int32)
    # voxel_size = np.array(voxel_size)
    # area_scope = np.array(area_scope)

    # voxelized_shape = ((area_scope[:,1] - area_scope[:,0])/voxel_size).astype(np.int32)
    # voxelized_shape[0], voxelized_shape[1]= voxelized_shape[1], voxelized_shape[0]
    
    # voxel_map = np.zeros(voxelized_shape, dtype=np.float32)
    # voxel_map[voxel_coords[:, 1], voxel_coords[:, 0], voxel_coords[:, 2]] = 1.0
    # # voxel_map = np.flip(np.flip(voxel_map, axis=0), axis=1)

    bev_map = voxel_map.sum(axis=2)
   
    bev_index = bev_map > 0
    bev_map = np.zeros([bev_map.shape[0], bev_map.shape[1], 3], dtype = np.uint8)
    bev_map[bev_index] = (228, 197, 85)
    
    classes = ['', 'Car', 'Pedestrian', 'Cyclist', 'Truck', 'Cone', 'Unknown', 'Dontcare']
    if gt_boxes is None or gt_boxes.shape[0] == 0:
        bev_image = cv2.resize(bev_map, dsize=(248, 216))
        bev_image = bev_image[::-1,:,:][:,::-1,:].transpose(1,0,2)
        return bev_image.copy()

    gt_boxes_corner3d = boxes3d_to_corners3d_lidar(gt_boxes)  # (N, 8, 3)
    gt_boxes_corners_bev = corners3d_to_bev_corners(gt_boxes_corner3d, voxel_size=voxel_size, area_scope=area_scope)    # (N, 4, 2)

    for k in range(gt_boxes_corners_bev.shape[0]):
        for j in range(0, 4):
            x1, y1 = gt_boxes_corners_bev[k, j, 0], gt_boxes_corners_bev[k, j, 1]
            x2, y2 = gt_boxes_corners_bev[k, (j + 1)%4, 0], gt_boxes_corners_bev[k, (j + 1) % 4, 1]
            cv2.line(bev_map, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
    

    # hard code
    # bev_image = bev_map.astype(np.uint8)
    bev_image = cv2.resize(bev_map, dsize=(248, 216))
    bev_image = bev_image[::-1,:,:][:,::-1,:].transpose(1,0,2)
    return bev_image.copy()