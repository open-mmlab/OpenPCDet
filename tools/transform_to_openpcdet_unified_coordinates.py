#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 15:39:50 2022

@author: yagmur


OpenPCDET unified coordinate system is the same with kitti. x -> front y->left z-> up

Nuscene x -> right y->front z-> up (should rotate 270 degree around z axis)
Nuscene max intensity : 255 min intensity : 0


"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn import preprocessing
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
    

def normalize_intensity(points, max_intensity, min_intensity):
    points[3] = (points[3] - min_intensity) / (max_intensity - min_intensity)
    return points




path = "/home/yagmur/Desktop/OpenPCDet/data/nuscenes/test/"
output_path = "/home/yagmur/Desktop/OpenPCDet/data/nuscenes/uniformed/"

NUSCENE_MAX_INTENSITY = 255
NUSCENE_MIN_INTENSITY = 0

if not os.path.exists(output_path):
    os.mkdir(output_path)
    
    
    
    
for lidar_file in glob.glob(path + '*.bin'):
    savename = lidar_file.split("/")[-1]
    lidar_data = (np.fromfile(str(lidar_file), dtype=np.float32)).reshape(-1,5)
    print(lidar_data[0]) # her .bin dosyasinin ilk lidar datasi
    save_uniformed = open(output_path+savename, "wb")
    uniformed_points = []
    for point in lidar_data:
      #  print(point)
        point = rotate_around_z(point,270)
        point = normalize_intensity(point,NUSCENE_MAX_INTENSITY,NUSCENE_MIN_INTENSITY)
        uniformed_points.append(point)
    print(uniformed_points[0])   # her .bin dosyasinin ilk lidar datasinin transform sonrasi hali
    save_uniformed.write(np.array(uniformed_points))


    
   