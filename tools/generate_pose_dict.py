#!/root/miniconda3/envs/pointpillars/bin/python
import sys
import json
import math
import numpy as np

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

nusc = NuScenes(version='v1.0-mini', dataroot='../data/nuscenes/v1.0-mini', verbose=True)
nusc.list_scenes()

# atan2 to quaternion:
# Quaternion(axis=[0, 0, 1], radians=atan2results)

def generate_pose_dict():
    global nusc
    token_to_cs_and_pose = {}

    for scene in nusc.scene:
        tkn = scene['first_sample_token']
        while tkn != "":
            #print('token:',tkn)
            sample = nusc.get('sample', tkn)
            #print('timestamp:', sample['timestamp'])
            sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            cs = nusc.get('calibrated_sensor',
                    sample_data['calibrated_sensor_token'])
            #print('calibrated sensor translation:', cs['translation'])
            #print('calibrated sensor rotation:', cs['rotation'])
            pose = nusc.get('ego_pose',
                sample_data['ego_pose_token'])
            #print('ego pose translation:', pose['translation'])
            #print('ego pose rotation:', pose['rotation'])
            token_to_cs_and_pose[tkn] = {
                    'timestamp' : sample['timestamp'],
                    'scene' : sample['scene_token'],
                    'cs_translation' : cs['translation'],
                    'cs_rotation' : cs['rotation'],
                    'ep_translation' : pose['translation'],
                    'ep_rotation' : pose['rotation'],
            }
            tkn = sample['next']

    print('Dict size:', sys.getsizeof(token_to_cs_and_pose)/1024/1024, ' MB')

    with open('token_to_pos.json', 'w') as handle:
        json.dump(token_to_cs_and_pose, handle, indent=4)


def generate_anns_dict():
    global nusc

    map_name_from_general_to_detection = {
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.wheelchair': 'ignore',
        'human.pedestrian.stroller': 'ignore',
        'human.pedestrian.personal_mobility': 'ignore',
        'human.pedestrian.police_officer': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'animal': 'ignore',
        'vehicle.car': 'car',
        'vehicle.motorcycle': 'motorcycle',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.truck': 'truck',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.emergency.ambulance': 'ignore',
        'vehicle.emergency.police': 'ignore',
        'vehicle.trailer': 'trailer',
        'movable_object.barrier': 'barrier',
        'movable_object.trafficcone': 'traffic_cone',
        'movable_object.pushable_pullable': 'ignore',
        'movable_object.debris': 'ignore',
        'static_object.bicycle_rack': 'ignore',
    }

    classes = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']

    token_to_anns = {}

    for scene in nusc.scene:
        tkn = scene['first_sample_token']
        print(scene['name'])
        categories_in_scene = set()
        while tkn != "":
            #print('token:',tkn)
            sample = nusc.get('sample', tkn)
            #print('timestamp:', sample['timestamp'])
            sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            cs = nusc.get('calibrated_sensor',
                    sample_data['calibrated_sensor_token'])
            #print('calibrated sensor translation:', cs['translation'])
            #print('calibrated sensor rotation:', cs['rotation'])
            pose = nusc.get('ego_pose',
                sample_data['ego_pose_token'])
            #print('ego pose translation:', pose['translation'])
            #print('ego pose rotation:', pose['rotation'])

            annos = np.zeros((len(sample['anns']),9))
            labels = []
            num_ignored = 0
            for i, anno_token in enumerate(sample['anns']):
                anno = nusc.get('sample_annotation', anno_token)
                cn = anno['category_name']
                name = map_name_from_general_to_detection[cn]
                if name == 'ignore':
                    num_ignored += 1
                    continue
                categories_in_scene.add(name)
                labels.append(classes.index(name)+1)
                #print(anno['category_name'])
                anno_vel = nusc.box_velocity(anno_token)
                box = Box(anno['translation'], anno['size'],
                    Quaternion(anno['rotation']), velocity=tuple(anno_vel))
                box.translate(-np.array(pose['translation']))
                box.rotate(Quaternion(pose['rotation']).inverse)
                box.translate(-np.array(cs['translation']))
                box.rotate(Quaternion(cs['rotation']).inverse)

                idx = i - num_ignored
                annos[idx, :3] = box.center
                annos[idx, 3] = box.wlh[1]
                annos[idx, 4] = box.wlh[0]
                annos[idx, 5] = box.wlh[2]
                r, x, y, z = box.orientation.elements
                annos[idx, 6] = 2. * math.atan2(math.sqrt(x*x+y*y+z*z),r)
                annos[idx, 7:] = box.velocity[:2] # this is actually global velocity
            annos = annos[:annos.shape[0]-num_ignored]

            labels = np.array(labels)
            indices = labels.argsort()
            labels.sort()
            annos = annos[indices]
            #print('Annos:\n', annos)
            token_to_anns[tkn] = {
                'pred_boxes': annos.tolist(),
                'pred_scores': [1.0] * annos.shape[0],
                'pred_labels': labels.tolist(),
            }
            tkn = sample['next']
        print(len(categories_in_scene), categories_in_scene)

    print('Dict size:', sys.getsizeof(token_to_anns)/1024/1024, ' MB')

    with open('token_to_anns.json', 'w') as handle:
        json.dump(token_to_anns, handle, indent=4)

generate_anns_dict()
#generate_pose_dict()
