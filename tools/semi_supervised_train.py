import argparse
from pathlib import Path
import random

import yaml
from easydict import EasyDict

from pcdet.datasets.neolix.neolix_pseudo_dataset import create_neolix_infos, NeolixDataset

def generate_list_file(data_path,list_file,suffix,org_list_file=''):
    data_list = ["pseudo/" + data.stem for data in Path(data_path).glob("*."+suffix)]

    if Path(org_list_file).exists():
        with open(org_list_file, "r") as lf:
            lines= lf.readlines()
        org_list = ["labeled/" + line.strip() for line in lines]
        data_list.extend(org_list)
    
    random.shuffle(data_list)
    with open(list_file, "w") as f:
        for data in data_list:
            f.write(data +"\n")

def make_dir(*dir_list):
    for dir_path in dir_list:
        Path(dir_path).mkdir(parents=True,exist_ok=True)

def rewrite_file_list(org_list_file, list_file, parent_folder):
    with open(org_list_file,"r") as f:
        val_list =[line.strip() for line in f.readlines()]

    with open(list_file, "w") as f:
        for data in val_list:
            f.write(parent_folder + "/" + data + "\n")


if __name__ == '__main__':

    rewrite_val_file = True
    use_npy_file = False

    root_path = "/nfs/neolix_data1/neolix_dataset/develop_dataset/pseudo_label_dataset/"
    pseudo_label_path = root_path + "training/label_2/pseudo"
    list_file = root_path + "ImageSets/train.txt"
    org_list_file = "/nfs/neolix_data1/neolix_dataset/develop_dataset/pseudo_label_dataset/ID_1022/ImageSets/train.txt"

    # generate train list
    make_dir(pseudo_label_path, Path(list_file).parent)
    if not Path(list_file).exists():
        print("generate list file: ", list_file)
        # generate_list_file(label_path, list_file, suffix="txt")
        generate_list_file(pseudo_label_path, list_file, suffix="txt", org_list_file=org_list_file)
    else:
        print("list file already exist in {}, make sure you don't have to change it.".format(list_file))

    # rewrite val.txt
    if rewrite_val_file:
        org_val_list_file = "/nfs/neolix_data1/neolix_dataset/develop_dataset/pseudo_label_dataset/ID_1022/ImageSets/val.txt"
        val_list_file = "/nfs/neolix_data1/neolix_dataset/develop_dataset/pseudo_label_dataset/ImageSets/val.txt"
        rewrite_file_list(org_val_list_file, val_list_file,"labeled")

    # import pdb; pdb.set_trace()
    # create infos pickle files
    cfg_file = "cfgs/dataset_configs/neolix_pseudo_dataset.yaml"
    dataset_cfg = EasyDict(yaml.load(open(cfg_file)))
    classes_list = ['Vehicle', 'Pedestrian', 'Cyclist', 'Unknown', 'Large_vehicle']
    #['Vehicle', 'Large_vehicle', 'Pedestrian', 'Cyclist', 'Bicycle', 'Unknown_movable', 'Unknown_unmovable']

    create_neolix_infos(
                                        dataset_cfg=dataset_cfg,
                                        class_names=classes_list,
                                        data_path=Path(root_path),
                                        save_path=Path(root_path),
                                        workers=8
                                        )

    # generate npy file
    if use_npy:
        # train npy
        train_dataset = NeolixDataset(dataset_cfg=dataset_cfg, class_names=classes_list, root_path=Path(root_path), training=True)
        processed_npy_path = root_path + f'process_data/{train_dataset.split}_process_results'
        train_dataset.process_results_dir = processed_npy_path
        print("process_data_path:", train_dataset.process_results_dir)
        train_dataset.__generateitem__(num_workers=4)
        # val npy
        # another method is to use set_split method of dataset class.
        val_dataset = NeolixDataset(dataset_cfg=dataset_cfg, class_names=classes_list, root_path=Path(root_path), training=False)
        processed_npy_path = root_path + f'process_data/{val_dataset.split}_process_results'
        val_dataset.process_results_dir = processed_npy_path
        print("process_data_path:", val_dataset.process_results_dir)
        val_dataset.__generateitem__(num_workers=4)




