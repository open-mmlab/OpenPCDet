#!/root/miniconda3/envs/pointpillars/bin/python
import sys
import json
import math
import copy
import random
import uuid
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

def gen_new_token(table_name):
    # Generate a unique anno token
    # each token is 32 chars
    global nusc
    
    while True:
        new_token = uuid.uuid4().hex
        if new_token not in nusc._token2ind[table_name]:
            nusc._token2ind[table_name][new_token] = -1 # enough for now
            break

    return new_token

# step defines the time between populated annotations in milliseconds
# step 50ms, 100ms, 150ms, ...
def populate_annos_v2(step):
    global nusc
    step = step//50
    scene_to_sd = {}
    scene_to_sd_cam = {}
    for i, sd_rec in enumerate(nusc.sample_data):
        for channel, dct in zip(['LIDAR_TOP', 'CAM_FRONT'], \
                [scene_to_sd, scene_to_sd_cam]):
            if sd_rec['channel'] == channel:
                scene_tkn = nusc.get('sample', sd_rec['sample_token'])['scene_token']
                if scene_tkn not in dct:
                    dct[scene_tkn] = []
                dct[scene_tkn].append(sd_rec)

    for dct in [scene_to_sd, scene_to_sd_cam]:
        for k, v in dct.items():
            dct[k] = sorted(v, key=lambda item: item['timestamp'])

    scene_to_kf_indexes = {}
    for k, v in scene_to_sd.items():
        # Filter based on time, also filter the ones which cannot
        # be interpolated
        is_kf_arr = [sd['is_key_frame'] for sd in v]
        kf_indexes = [i for i in range(len(is_kf_arr)) if is_kf_arr[i]]
        scene_to_kf_indexes[k] = kf_indexes
    
    all_new_sample_datas = []
    all_new_samples = []
    all_new_annos = []
    for scene in nusc.scene:
        print('Processing scene', scene['name'])
        sd_records = scene_to_sd[scene['token']]
        sd_records_cam = scene_to_sd_cam[scene['token']]
        kf_indexes = scene_to_kf_indexes[scene['token']]
        for idx in range(len(kf_indexes) - 1):
            # generate sample between these two
            begin_kf_idx = kf_indexes[idx]
            end_kf_idx = kf_indexes[idx+1]
            cur_sample = nusc.get('sample', sd_records[begin_kf_idx]['sample_token'])
            next_sample = nusc.get('sample', sd_records[end_kf_idx]['sample_token'])
            # if these two are equal, this is a problem for interpolation
            assert cur_sample['token'] != next_sample['token']
            sd_rec_indexes = np.arange(begin_kf_idx+step, end_kf_idx-step+1, step)

            new_samples = []
            new_sample_annos = []
            for sd_rec_idx in sd_rec_indexes:
                sd_rec = sd_records[sd_rec_idx]
                new_token = gen_new_token('sample')
                # find the sd_record_cam with closest timestamp
                lidar_ts = sd_rec['timestamp']
                cam_ts_arr = np.asarray([sd_rec_cam['timestamp'] \
                        for sd_rec_cam in sd_records_cam])
                cam_idx = (np.abs(cam_ts_arr - lidar_ts)).argmin()
                sd_rec_cam = sd_records_cam[cam_idx]
                new_samples.append({
                        'token': new_token,
                        'timestamp' : lidar_ts,
                        'prev': "",
                        'next': "",
                        'scene_token': scene['token'],
                        'data': {'LIDAR_TOP': sd_rec['token'],
                            'CAM_FRONT': sd_rec_cam['token']},
                        'anns': [],
                })

                # update sample data record
                sd_rec['sample_token'] = new_samples[-1]['token']
                sd_rec['is_key_frame'] = True # not sure this is right
                if not sd_rec_cam['is_key_frame']:
                    sd_rec_cam['sample_token'] = new_samples[-1]['token']
                    sd_rec_cam['is_key_frame'] = True # not sure this is right
                else:
                    # Fabricate an sd_rec_cam with a new token
                    # because we cannot override this one as it is a keyframe
                    new_sd_rec_cam = copy.deepcopy(sd_rec_cam)
                    new_token = gen_new_token('sample_data')
                    new_sd_rec_cam['token'] = new_token
                    new_sd_rec_cam['sample_token'] = new_samples[-1]['token'] 
                    # I am not sure whether this one should be befor or after
                    # sd_rec_cam, but I will assume it will be after
                    new_sd_rec_cam['prev'] = sd_rec_cam['token']
                    new_sd_rec_cam['next'] = sd_rec_cam['next']
                    if new_sd_rec_cam['next'] != "":
                        nusc.get('sample_data', new_sd_rec_cam['next'])['prev'] = \
                                new_token
                    sd_rec_cam['next'] = new_token

                    # Do I need to generate a corresponding ego_pose_rec? I hope not
                    all_new_sample_datas.append(new_sd_rec_cam)

            # link the samples
            if not new_samples:
                continue

            cur_sample['next'] = new_samples[0]['token']
            assert cur_sample['timestamp'] < new_samples[0]['timestamp']
            new_samples[0]['prev'] = cur_sample['token']
            for i in range(1, len(new_samples)):
                new_samples[i-1]['next'] = new_samples[i]['token']
                new_samples[i]['prev'] = new_samples[i-1]['token']
            new_samples[-1]['next'] = next_sample['token']
            next_sample['prev'] = new_samples[-1]['token']

            # Generate annotations
            # For each anno in the cur_sample, find its corresponding anno
            # in the next sample. The matching can be done via instance_token
            total_time_diff = next_sample['timestamp'] - cur_sample['timestamp']
            for cur_anno_tkn in cur_sample['anns']:
                cur_anno = nusc.get('sample_annotation', cur_anno_tkn)
                next_anno_tkn = cur_anno['next']
                if next_anno_tkn == "":
                    continue
                next_anno = nusc.get('sample_annotation', next_anno_tkn)

                new_annos = []
                # Interpolate this anno for all new samples
                for new_sample in new_samples:
                    new_token = gen_new_token('sample_annotation')
                    new_anno = copy.deepcopy(cur_anno)

                    new_anno['token'] = new_token
                    new_anno['sample_token'] = new_sample['token']
                    new_sample['anns'].append(new_token)

                    time_diff = new_sample['timestamp'] - cur_sample['timestamp']
                    rratio = time_diff / total_time_diff
                    new_anno['translation'] = (1.0 - rratio) * \
                            np.array(cur_anno['translation'], dtype=float) + \
                            rratio * np.array(next_anno['translation'], dtype=float)
                    new_anno['translation'] = new_anno['translation'].tolist()
                    new_anno['rotation'] = Quaternion.slerp(
                            q0=Quaternion(cur_anno['rotation']),
                            q1=Quaternion(next_anno['rotation']),
                            amount=rratio
                    ).elements.tolist()
                    new_anno['prev'] = ''
                    new_anno['next'] = ''
                    new_annos.append(new_anno)

                # link the annos
                cur_anno['next'] = new_annos[0]['token']
                new_annos[0]['prev'] = cur_anno_tkn
                for i in range(1, len(new_annos)):
                    new_annos[i-1]['next'] = new_annos[i]['token']
                    new_annos[i]['prev'] = new_annos[i-1]['token']
                new_annos[-1]['next'] = next_anno_tkn
                next_anno['prev'] = new_annos[-1]['token']

                all_new_annos.extend(new_annos)
                # increase the number of annos in the instance table
                nusc.get('instance', cur_anno['instance_token'])['nbr_annotations'] += \
                        len(new_annos)

            all_new_samples.extend(new_samples)

            scene['nbr_samples'] += len(new_samples)

    nusc.sample.extend(all_new_samples)
    nusc.sample_annotation.extend(all_new_annos)
    nusc.sample_data.extend(all_new_sample_datas)

    # Dump the modified scene, sample, sample_data, sample_annotations, and instance tables
    indent_num=0
    print('Dumping the tables')
    with open('scene.json', 'w') as handle:
        json.dump(nusc.scene, handle, indent=indent_num)
    
    for sd in nusc.sample:
        del sd['anns']
        del sd['data']
    with open('sample.json', 'w') as handle:
        json.dump(nusc.sample, handle, indent=indent_num)

    for sd in nusc.sample_data:
        del sd['sensor_modality']
        del sd['channel']
    with open('sample_data.json', 'w') as handle:
        json.dump(nusc.sample_data, handle, indent=indent_num)

    for sd in nusc.sample_annotation:
        del sd['category_name']
    with open('sample_annotation.json', 'w') as handle:
        json.dump(nusc.sample_annotation, handle, indent=indent_num)

    with open('instance.json', 'w') as handle:
        json.dump(nusc.instance, handle, indent=indent_num)

def main():
    if len(sys.argv) == 3 and sys.argv[1] == 'populate_annos':
        step = int(sys.argv[2])
        populate_annos_v2(step)
    elif len(sys.argv) == 2 and sys.argv[1] == 'generate_dicts':
        generate_anns_dict()
        generate_pose_dict()
    else:
        print('Usage error, doing nothing.')

if __name__ == "__main__":
    main()
