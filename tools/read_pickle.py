#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 10:12:52 2022

@author: yagmur

with (open("/home/yagmur/OpenPCDet/data/kitti/kitti_infos_test.pkl", "rb")) 
 #with (open("epoch_200/val/result.pkl", "rb")) as openfile:
     
"""

import pickle

objects = []
with (open("/home/yagmur/Downloads/PANDASET/001/lidar/00.pkl", "rb")) as openfile:  #with (open("epoch_200/val/result.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
for i in objects:        
    print(i)   


print("result from", str(len(objects[0])), "images")     