import argparse
import os
from pathlib import Path
import random

import yaml
import numpy as np
import pickle
from easydict import EasyDict
from tqdm import tqdm
import concurrent.futures as futures

from pcdet.datasets.neolix.neolix_pseudo_dataset import create_neolix_infos, NeolixDataset

def generate_list_file(data_path,list_file,suffix,org_list_file=''):
    """Generate new list file for labeled data and pseudo-labeled data.
    
    Args:
      data_path: path to pseudo labels.
      list_file: path to save new training list.
      org_list_file: original training list file for labeled data.
    Returns:
      
    """
    data_list = ["pseudo/" + data.stem for data in Path(data_path).glob("*."+suffix)]

    if Path(org_list_file).exists():
        with open(org_list_file, "r") as lf:
            lines= lf.readlines()
        org_list = ["labeled/" + line.strip() for line in lines]
        # data_list = random.sample(data_list, len(org_list))
        data_list.extend(org_list)
    
    random.shuffle(data_list)
    with open(list_file, "w") as f:
        for data in data_list:
            f.write(data +"\n")

def make_dir(*dir_list):
    for dir_path in dir_list:
        Path(dir_path).mkdir(parents=True,exist_ok=True)

def rewrite_file_list(org_list_file, list_file, parent_folder):
    """Add parent_folder to paths in list file. 
    
    Args:
      org_list_file: original list file.
      list_file: path for saving new list file.
      parent_folder: parent folder to be added.
    Returns:

    """

    with open(org_list_file,"r") as f:
        val_list =[line.strip() for line in f.readlines()]

    with open(list_file, "w") as f:
        for data in val_list:
            f.write(parent_folder + "/" + data + "\n")

def parse_args():
    parser = argparse.ArgumentParser(description="argument parser")
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--root_path', type=str, default=None, help='specify the root path for generated dataset, e.g. path/to/pseudo_label_dataset')
    parser.add_argument('--org_list_file', type=str, default=None, help='specify the orginal list file to create new train list file')
    parser.add_argument('--org_val_list_file', type=str, default=None, help='specify the original val list file to add sub folder' )
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the dataset config file to create infos' )
    parser.add_argument('--use_npy_file', type=str, default=False, help='whether to use offline-agumented npy file)
    parser.add_argument('--check_empty_label', default=False, help='whether to check empty labels')
    parser.add_argument('--need_count_labels', default=True, help='whether to count anchor size in labels')
    parser.add_argument('--train_cfg_file', default="cfgs/neolix_models/pointpillar_1029.yaml", help='specify the yaml file for training')
    parser.add_argument('--workers', type=int, default=4, help='number of multi thread workers')
    args = parser.parse_args()

    return args

def count_labels(l_path, selected_class,num_workers=4):
    """Calculate anchor size.
    Args:
      l_path: path to labels.
      selected_class: class to be calculated.
    Returns:
      (w, l, h)
    """
    def count_single_label(label_file):
        size = []
        z_axis = []
        num = 0
        with open(label_file,"r") as f:
            label_lines = f.readlines()
        for label_line in label_lines:
            label_line = label_line.split(" ")
            if label_line[0] == selected_class:
                num += 1
                size.append([float(label_line[8]), float(label_line[9]), float(label_line[10])])
                z_axis.append(float(label_line[13]))
        np_size = np.array(size)
        np_z_axis = np.array(z_axis)
        if np_size.shape[0] == 0:
            return 0,0,0,0,0
        s_h = np_size[:, 0].sum()
        s_w = np_size[:, 1].sum()
        s_l = np_size[:, 2].sum()
        s_z = np_z_axis.sum()
        return s_h, s_w, s_l, s_z, num
    label_list = list(Path(l_path).glob("**/*.txt"))
    sum_h = 0
    sum_w = 0
    sum_l = 0
    sum_z = 0
    total_num = 0
    with futures.ThreadPoolExecutor(num_workers) as executor:
        for result in executor.map(count_single_label, label_list):
            # print(result)
            sum_h += result[0]
            sum_w += result[1]
            sum_l += result[2]
            sum_z += result[3]
            total_num += result[4]
    avg_h = sum_h / total_num
    avg_w = sum_w / total_num
    avg_l = sum_l / total_num
    avg_z = sum_z / total_num
    print("the mean height of %s" % selected_class, avg_h)
    print("the mean width of %s" % selected_class, avg_w)
    print("the mean length of %s" % selected_class, avg_l)
    print("the mean z coordinate of %s" % selected_class, avg_z)
    return [round(avg_w,2), round(avg_l,2), round(avg_h,2)]

def org_count_labels(l_path, selected_class):
    size = []
    z_axis = []
    label_list = list(Path(l_path).glob("**/*.txt"))
    for label_file in tqdm(label_list):
        with open(label_file,"r") as f:
            label_lines = f.readlines()
        for label_line in label_lines:
            label_line = label_line.split(" ")
            if label_line[0] == selected_class:
                size.append([float(label_line[8]), float(label_line[9]), float(label_line[10])])
                z_axis.append(float(label_line[13]))
            # if (float(label_line[8])>3)|(float(label_line[9])>3)|(float(label_line[10])>3):
            #     print("the vehicle is in %s" % l_path+l)
    np_size = np.array(size)
    np_z_axis = np.array(z_axis)
    h = np_size[:, 0].mean()
    w = np_size[:, 1].mean()
    l = np_size[:, 2].mean()
    # print("the number of %s is %d" % (selected_class, np_size.shape[0]))
    print("the mean height of %s" % selected_class, h)
    print("the mean width of %s" % selected_class, w)
    print("the mean length of %s" % selected_class, l)
    print("the mean z coordinate of %s" % selected_class, np_z_axis.mean())
    return [round(w,2), round(l,2), round(h,2)]


if __name__ == '__main__':

    args = parse_args()
    # args.cfg_file = "cfgs/dataset_configs/neolix_pseudo_dataset.yaml"
    classes_list = ['Vehicle', 'Pedestrian', 'Cyclist', 'Unknown', 'Large_vehicle']

    root_path = args.root_path#"/nfs/neolix_data1/neolix_dataset/develop_dataset/pseudo_label_dataset"
    pseudo_label_path = root_path + "/training/label_2/pseudo"
    # args.org_list_file = "/nfs/neolix_data1/neolix_dataset/develop_dataset/lidar_object_detection/ID_1022/ImageSets/train.txt"

    # generate train list
    if args.org_list_file:
        list_file = root_path + "/ImageSets/train.txt"
        make_dir(pseudo_label_path, Path(list_file).parent)
        print("generate list file: ", list_file)
        generate_list_file(pseudo_label_path, list_file, suffix="txt", org_list_file=args.org_list_file)

    # rewrite val.txt
    if args.org_val_list_file:
        # org_val_list_file = "/nfs/neolix_data1/neolix_dataset/develop_dataset/lidar_object_detection/ID_1022/ImageSets/val.txt"
        val_list_file = root_path + "/ImageSets/val.txt"
        rewrite_file_list(args.org_val_list_file, val_list_file, parent_folder="labeled")

    # create infos pickle files
    if args.cfg_file:
        dataset_cfg = EasyDict(yaml.load(open(args.cfg_file)))
        create_neolix_infos(
                                            dataset_cfg=dataset_cfg,
                                            class_names=classes_list,
                                            data_path=Path(root_path),
                                            save_path=Path(root_path),
                                            workers=args.workers
                                            )

    # generate npy file
    if args.use_npy_file:
        # train npy
        train_dataset = NeolixDataset(dataset_cfg=dataset_cfg, class_names=classes_list, root_path=Path(root_path), training=True)
        processed_npy_path = root_path + f'/process_data/{train_dataset.split}_process_results'
        train_dataset.process_results_dir = processed_npy_path
        print("process_data_path:", train_dataset.process_results_dir)
        train_dataset.__generateitem__(num_workers=args.workers)
        # val npy
        # another method is to use set_split method of dataset class.
        val_dataset = NeolixDataset(dataset_cfg=dataset_cfg, class_names=classes_list, root_path=Path(root_path), training=False)
        processed_npy_path = root_path + f'/process_data/{val_dataset.split}_process_results'
        val_dataset.process_results_dir = processed_npy_path
        print("process_data_path:", val_dataset.process_results_dir)
        val_dataset.__generateitem__(num_workers=args.workers)

    if args.check_empty_label:
        file_list = list(Path(pseudo_label_path).glob("*.txt"))
        empty_list = root_path + "/empty_pseudo_list.txt"
        with open(empty_list, "w") as ef:
            for i in tqdm(range(len(file_list))):
                pseudo_label = file_list[i]
                with open(pseudo_label, "r") as f:
                    content = f.readlines()
                if len(content) < 1:
                    print(content)
                    ef.write(pseudo_label.name+"\n")

    # change training configs: anchor
    if args.need_count_labels:
        label_path = root_path + "/training/label_2"
        # train_cfg_file = "cfgs/neolix_models/pointpillar_1029.yaml"

        with open(train_cfg_file, "r") as yaml_file:
            train_cfg = yaml_file.read()
            yaml_file.seek(0)
            train_cfg_yaml = yaml.load(yaml_file)

        for i,c in enumerate(class_list):
            assert train_cfg_yaml["MODEL"]["DENSE_HEAD"]["ANCHOR_GENERATOR_CONFIG"][i]['class_name'] == c
            org_str = str(train_cfg_yaml["MODEL"]["DENSE_HEAD"]["ANCHOR_GENERATOR_CONFIG"][i]['anchor_sizes'][0])
            if train_cfg.find(org_str) != -1:
                new_str = str(count_labels(class_path, c, num_workers=args.workers))
                # new_str = str(org_count_labels(class_path, c))
                train_cfg = train_cfg.replace(org_str, new_str)
            else:
                print(f"************{c} not find***************")
        with open(args.train_cfg_file, "w") as yaml_file:
            yaml_file.write(train_cfg)