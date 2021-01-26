import numpy as np
import pickle

def merge_cls():
    """
    sub_class: ['car','truck', 'construction_vehicle', 'bus', 'trailer',
              'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    main_class: ['Vehicle', 'Pedestrian', 'Cyclist', 'Unknown']


    sub_class               main_class
            car       |     car
    __________________|___________________________
            truck     |     car
    __________________|___________________________
  construction_vehicle|     car
    __________________|___________________________
            bus       |     car
    __________________|___________________________
          trailer     |     car
    __________________|___________________________
          motorcycle  |      bicycle
    __________________|___________________________
           bicycle    |      bicycle
    __________________|___________________________
           pedestrian |    pedestrian
    __________________|___________________________
          barrier     |     barrier
    __________________|___________________________
        traffic_cone  |     barrier
    __________________|___________________________
    """
    train_pkl_path = "/nfs/nas/datasets/nuScenes/nuscenes_datasets/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_train.pkl"
    val_pkl_path = "/nfs/nas/datasets/nuScenes/nuscenes_datasets/nuscenes/v1.0-trainval/nuscenes_infos_10sweeps_val.pkl"
    gtdb_info_path = ""
    gt_folder_path = ""
    pkl_class_merge(train_pkl_path)
    pkl_class_merge(val_pkl_path)
    # gt_pkl_class_merge(gtdb_info_path)
    # gt_folder_class_merge(gt_folder_path)




def pkl_class_merge(pkl_path):
    f = open(pkl_path, 'rb')
    infos = pickle.load(f)
    sub_cls_2_main_cls = {'car': 'car', 'truck': 'car', 'construction_vehicle': 'car', 'bus': 'car', 'trailer': 'car',
                          'motorcycle': 'bicycle', 'bicycle': 'bicycle',
                          'pedestrian': 'pedestrian',
                          'barrier': 'barrier', 'traffic_cone': 'barrier',
                          'ignore': 'ignore'}
    new_infos = []
    for info in infos:
        new_names = []
        for gt_name in info['gt_names'].tolist():
            new_names.append(sub_cls_2_main_cls[gt_name])
        info['gt_names'] = np.array(new_names)
        new_infos.append(info)
    with open(pkl_path.strip(".pkl")+"_new.pkl", 'wb') as f:
        pickle.dump(new_infos, f)


def gt_pkl_class_merge(gt_pkl_path):
    pass


def gt_folder_class_merge(gt_folder_path):
    pass


merge_cls()