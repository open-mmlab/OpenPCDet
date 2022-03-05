#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 15:35:37 2022

@author: yagmur

1 scene contains almost 40 samples (40 frames)

1 sample contains prev - nevt frame keys, anns key for all the annotations in this scene, lidar data token and cams token etc

http://sayef.tech/post/introduction-to-nuscenes-dataset-for-autonomous-driving/ 

https://scale.com/open-datasets/nuscenes/tutorial

"""


from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-mini', dataroot='/home/yagmur/Downloads/Nuscene', verbose=True)
scene = nusc.scene[0]
#print(scene)

# sample = nusc.get('sample', scene['first_sample_token'])

#print(sample)
#first_annotation = nusc.get('sample_annotation', sample['anns'][0])
#print(first_annotation)

# lidar_data = sample['data']['LIDAR_TOP']
# lidar_data_name = nusc.get('sample_data', lidar_data)['filename']
# print(lidar_data_name)
# print(lidar_data) #9d9bf11fb0e144c8b446d54a8a00184f

# cam_back_data = sample['data']['CAM_BACK']
# print(cam_back_data) #03bea5763f0f4722933508d5999c5fd8

# nusc.render_sample_data(lidar_data)
# nusc.render_sample_data(cam_back_data)

# next_sample = sample['next']
# sample = nusc.get('sample', next_sample)
# print(sample)

sample = nusc.get('sample', scene['first_sample_token'])
first_annotation = nusc.get('sample_annotation', sample['anns'][0])

n = 3 #(10 sample will be collected for test)

while n:
    lidar_data = sample['data']['LIDAR_TOP']
    lidar_data_name = nusc.get('sample_data', lidar_data)['filename']
    print(lidar_data_name)
    nusc.render_sample_data(lidar_data)
    next_sample = sample['next']
    sample = nusc.get('sample', next_sample)
    n -= 1

    current_annotation = nusc.get('sample_annotation', sample['anns'][0])
    length_annotation = len(sample['anns'])
 #   print("length anno",length_annotation)
    
    for i in range(length_annotation):
        print(current_annotation['category_name'])
        current_annotation =  nusc.get('sample_annotation', sample['anns'][i])
        
