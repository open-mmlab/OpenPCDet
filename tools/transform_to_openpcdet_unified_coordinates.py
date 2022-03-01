#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 15:39:50 2022

@author: yagmur


OpenPCDET unified coordinate system is the same with kitti. x -> front y->left z-> up

Nuscene x -> right y->front z-> up (should rotate 270 degree around z axis)


"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import glob

def rotate_around_z(points, angle):
   # print(points)
    rotation_radians = np.radians(angle)
    rotation_axis = np.array([0, 0, 1]) # z
    rotation_vector = rotation_radians * rotation_axis
    rotation = R.from_rotvec(rotation_vector)
    points[:3] = rotation.apply(points[:3])
    return points
 #   print(points) 
    

path = "/home/yagmur/Desktop/OpenPCDet/data/lyft/test/"
output_path = "/home/yagmur/Desktop/OpenPCDet/data/lyft/uniformed/"

if not os.path.exists(output_path):
    os.mkdir(output_path)
    
    
    
    
for lidar_file in glob.glob(path + '*.bin'):
    savename = lidar_file.split("/")[-1]
    lidar_data = (np.fromfile(str(lidar_file), dtype=np.float32)).reshape(-1,5)
    print(lidar_data[0])
    save_uniformed = open(output_path+savename, "wb")
    uniformed_points = []
    for point in lidar_data:
        uniformed_points.append(rotate_around_z(point,270))
    print(uniformed_points[0])  
    save_uniformed.write(np.array(uniformed_points))


    
   