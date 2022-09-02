import numpy as np
import pickle as pkl
from pathlib import Path
import tqdm
import copy
import os 


def create_integrated_db_with_infos(args, root_path):
    """
    Args:
        args:
    Returns:

    """
    # prepare
    db_infos_path = args.src_db_info
    db_info_global_path = db_infos_path
    global_db_path = root_path / (args.new_db_name + '.npy')

    db_infos = pkl.load(open(db_infos_path, 'rb'))
    db_info_global = copy.deepcopy(db_infos)
    start_idx = 0
    global_db_list = []

    for category, class_info in db_infos.items():
        print('>>> Start processing %s' % category)
        for idx, info in tqdm.tqdm(enumerate(class_info), total=len(class_info)):
            obj_path = root_path / info['path']
            obj_points = np.fromfile(str(obj_path), dtype=np.float32).reshape(
                [-1, args.num_point_features])
            num_points = obj_points.shape[0]
            if num_points != info['num_points_in_gt']:
                obj_points = np.fromfile(str(obj_path), dtype=np.float64).reshape([-1, args.num_point_features])
                num_points = obj_points.shape[0]
                obj_points = obj_points.astype(np.float32)
            assert num_points == info['num_points_in_gt']
                
            db_info_global[category][idx]['global_data_offset'] = (start_idx, start_idx + num_points)
            start_idx += num_points
            global_db_list.append(obj_points)

    global_db = np.concatenate(global_db_list)

    with open(global_db_path, 'wb') as f:
        np.save(f, global_db)

    with open(db_info_global_path, 'wb') as f:
        pkl.dump(db_info_global, f)

    print(f"Successfully create integrated database at {global_db_path}")
    print(f"Successfully create integrated database info at {db_info_global_path}")

    return db_info_global, global_db


def verify(info, whole_db, root_path, num_point_features):
    obj_path = root_path / info['path']
    obj_points = np.fromfile(str(obj_path), dtype=np.float32).reshape([-1, num_point_features])
    mean_origin = obj_points.mean()

    start_idx, end_idx = info['global_data_offset']
    obj_points_new = whole_db[start_idx:end_idx]
    mean_new = obj_points_new.mean()

    assert mean_origin == mean_new

    print("Verification pass!")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--src_db_info', type=str, default='../../data/waymo/waymo_processed_data_v0_5_0_waymo_dbinfos_train_sampled_1_multiframe_-4_to_0_tail_parallel.pkl', help='')
    parser.add_argument('--new_db_name', type=str, default='waymo_processed_data_v0_5_0_gt_database_train_sampled_1_multiframe_-4_to_0_tail_parallel_global', help='')
    parser.add_argument('--num_point_features', type=int, default=6, help='number of feature channels for points')
    parser.add_argument('--class_name', type=str, default='Vehicle', help='category name for verification')

    args = parser.parse_args()

    root_path = Path(os.path.dirname(args.src_db_info))

    db_infos_global, whole_db = create_integrated_db_with_infos(args, root_path)
    # simple verify
    verify(db_infos_global[args.class_name][0], whole_db, root_path, args.num_point_features)
