import argparse
import os
from pathlib import Path
import random

import yaml
import shutil
import numpy as np
import pickle
from easydict import EasyDict
from tqdm import tqdm
import concurrent.futures as futures
import subprocess
import torch
from torch.utils.data import DataLoader

from pcdet.datasets.neolix.neolix_pseudo_dataset import create_neolix_infos, NeolixDataset

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils, object3d_neolix, box_utils
from demo import DemoDataset

def generate_pseudo_labels(args, cfg):
    logger = common_utils.create_logger()
    logger.info('-----------------Generating Pseudo Labels-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.org_raw_data_path), ext=args.ext, logger=logger, label_path=None
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    dataloader = DataLoader(
        demo_dataset, batch_size=args.infer_bs, pin_memory=True, num_workers=args.infer_workers,
        shuffle=False, collate_fn=demo_dataset.collate_batch,
        drop_last=False, timeout=0
    )
    
    with torch.no_grad():
        for idx, data_dict in enumerate(dataloader):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            load_data_to_gpu(data_dict)
            pred_dicts, ret_dict = model(data_dict)
            annos = demo_dataset.generate_prediction_dicts(
                data_dict, pred_dicts, cfg.CLASS_NAMES,
                output_path=Path(args.root_path) / "training/label_2/pseudo"
            )

    logger.info('Generating Pseudo Labels done.')


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
        #data_list = random.sample(data_list, len(org_list))
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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description="argument parser")
    parser.add_argument('--teacher_cfg_file', type=str, default=None, help='specify the config of pretrained teacher network for pseudo labels')
    parser.add_argument('--gpus', type=str, default="0", help='specify the gpus that can be used for inference, e.g. 1,2')
    parser.add_argument('--infer_bs', type=int, default=12, help='specify the batch size used for inference')
    parser.add_argument('--infer_workers', type=int, default=4, help='specify the woekers used for inference')
    parser.add_argument('--score_thresh', type=float, default=None, help='specify the score thresh for generating pseudo labels')
    # parser.add_argument('--data_path', type=str, default='demo_data', help='specify the raw point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    # parser.add_argument('--label_path', type=str, default=None, help='specify the point cloud data label or directory')
    # parser.add_argument('--save_path', type=str, default=None, help='specify path to save pseudo labels')
    # parser.add_argument('--org_root_path', type=str, default=None, help='specify the original root path for training dataset(please not end with "/")')
    parser.add_argument('--copy_org_data', type=str2bool, default=True, help='whether to copy labeled data and raw velodyne')
    parser.add_argument('--org_velodyne_path', type=str, default=None, help='specify the velodyne path of original labeled data(please not end with "/")')
    parser.add_argument('--org_label_path', type=str, default=None, help='specify the label path of original labeled data(please not end with "/")')
    parser.add_argument('--org_raw_data_path', type=str, default=None, help='specify the path of raw velodyne without label(please not end with "/")')

    parser.add_argument('--root_path', type=str, default=None, help='specify the root path for generated dataset(please not end with "/"), e.g. path/to/pseudo_label_dataset')
    parser.add_argument('--org_list_file', type=str, default=None, help='specify the orginal list file to create new train list file')
    parser.add_argument('--org_val_list_file', type=str, default=None, help='specify the original val list file to add sub folder' )
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the dataset config file to create infos' )
    parser.add_argument('--use_npy_file', type=str, default=False, help='whether to use offline-agumented npy file')
    parser.add_argument('--check_empty_label', default=False, help='whether to check empty labels')
    parser.add_argument('--need_count_labels', default=True, help='whether to count anchor size in labels')
    parser.add_argument('--train_cfg_file', default=None, help='specify the yaml file for training')
    parser.add_argument('--workers', type=int, default=4, help='number of multi thread workers')
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    import time
    start_time = time.time()
    args = parse_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    teacher_cfg = cfg_from_yaml_file(args.teacher_cfg_file, cfg)
    classes_list = ['Vehicle', 'Pedestrian', 'Cyclist', 'Unknown', 'Large_vehicle']
    print("="*10 + f"{len(classes_list)} classes are used: {classes_list}" + "="*10)

    root_path = args.root_path
    make_dir(root_path)
    pseudo_label_path = root_path + "/training/label_2/pseudo"

    # generating pseudo labels.
    if args.copy_org_data:
        print(f"Original training data will be automatically copyed to {root_path}.")
        # make_dir(root_path+"/training/velodyne/labeled", root_path+"/training/label_2/labeled", root_path+"/training/velodyne/pseudo")
        try:
            # python <3.8: dst in copytree must not already exist, otherwise error will occured; python>3.8: parameter dirs_exist_ok is added to avoid error.
            shutil.copytree(args.org_velodyne_path, args.root_path + "/training/velodyne/labeled")
            shutil.copytree(args.org_label_path, args.root_path + "/training/label_2/labeled")
            shutil.copytree(args.org_raw_data_path, args.root_path + "/training/velodyne/pseudo")
            # subprocess.run(f"cp -r {args.org_velodyne_path}/* {root_path}/training/velodyne/labeled", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # subprocess.run(f"cp -r {args.org_label_path}/* {root_path}/training/label_2/labeled", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # subprocess.run(f"cp -r {args.org_raw_data_path}/* {root_path}/training/velodyne/pseudo", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # subprocess.run(f'find {args.org_raw_data_path} -name "*{args.ext}" | xargs -i cp {{}} {root_path}/training/velodyne/pseudo', shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            print(f"Failed to copy file{e}.")
            raise e
        else:
            print("copy original training data successfully!")
    
    mark_time1 = time.time()

    if args.score_thresh:
        teacher_cfg.MODEL.POST_PROCESSING.SCORE_THRESH = args.score_thresh
    print(f"generating pseudo labels with score thresh {teacher_cfg.MODEL.POST_PROCESSING.SCORE_THRESH}")
    generate_pseudo_labels(args, teacher_cfg)

    mark_time2 = time.time()

    # generate train list
    if args.org_list_file:
        list_file = root_path + "/ImageSets/train.txt"
        make_dir(pseudo_label_path, Path(list_file).parent)
        print("generate list file: ", list_file)
        generate_list_file(pseudo_label_path, list_file, suffix="txt", org_list_file=args.org_list_file)

    # rewrite val.txt
    if args.org_val_list_file:
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

    mark_time3 = time.time()

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
        assert args.train_cfg_file is not None
        with open(args.train_cfg_file, "r") as yaml_file:
            train_cfg = yaml_file.read()
            yaml_file.seek(0)
            train_cfg_yaml = yaml.load(yaml_file)

        for i,c in enumerate(classes_list):
            assert train_cfg_yaml["MODEL"]["DENSE_HEAD"]["ANCHOR_GENERATOR_CONFIG"][i]['class_name'] == c
            org_str = str(train_cfg_yaml["MODEL"]["DENSE_HEAD"]["ANCHOR_GENERATOR_CONFIG"][i]['anchor_sizes'][0])
            if train_cfg.find(org_str) != -1:
                new_str = str(count_labels(label_path, c, num_workers=args.workers))
                train_cfg = train_cfg.replace(org_str, new_str)
            else: 
                print(f"************{c} not find***************")
        with open(args.train_cfg_file, "w") as yaml_file:
            print(f"====================change anchor size in config file: {args.train_cfg_file}====================")
            yaml_file.write(train_cfg)

    end_time = time.time()
    total_time = end_time - start_time
    print("time for copy:",mark_time1 - start_time)
    print("time for inference:", mark_time2 - mark_time1)
    print("time for infos:", mark_time3 - mark_time2)
    print(f"total time: {total_time}({total_time//3600}h,{total_time%3600//60 }m, {3600%60}s)")