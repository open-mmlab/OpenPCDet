#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 10:47:57 2022

@author: yagmur
"""
import numpy as np
import os

""" kitti """

lidar_file = "/home/yagmur/OpenPCDet/data/kitti/training/velodyne/000001.bin"
assert os.path.isfile(lidar_file)
lidar_data = np.fromfile(str(lidar_file), dtype=np.float32)  
print("kitti .bin lidar data shape", lidar_data.shape)

print("one kitti lidar data", lidar_data.reshape(-1,4))

""" lyft """

lidar_file = "/home/yagmur/Downloads/lyft/train/train_lidar/host-a004_lidar1_1232815252301696606.bin"
assert os.path.isfile(lidar_file)
lidar_data = np.fromfile(str(lidar_file), dtype=np.float32)
print("one lyft lidar data", lidar_data.reshape(-1,5)[:, :4])
print("lyft .bin lidar data shape", lidar_data.shape)


""" nuscene """

lidar_file = "/home/yagmur/Downloads/Nuscene/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin"
assert os.path.isfile(lidar_file)
lidar_data = np.fromfile(str(lidar_file), dtype=np.float32)
print("one nuscene lidar data", lidar_data.reshape(-1,5))
print("nuscene .bin lidar data shape", lidar_data.shape)


""" 
nuscene ve lyft te son bilgi ne bilmiyorum ama openpcdette point clouds okurken son veriyi atmislar.

print("one lyft lidar data", lidar_data.reshape(-1,5))[:, :4]  seklinde


"""
