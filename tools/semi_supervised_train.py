import argparse
from pathlib import Path
import random

import yaml
from easydict import EasyDict

from pcdet.datasets.neolix.neolix_dataset import create_neolix_infos

def generate_list_file(data_path,list_file,suffix,org_list_file=''):
    data_list = list(Path(data_path).glob("*."+suffix))

    if Path(org_list_file).exists():
        with open(org_list_file, "r") as lf:
            lines= lf.readlines()
        org_list = [Path(line.strip()) for line in lines]
        data_list.extend(org_list)
    
    random.shuffle(data_list)
    with open(list_file, "w") as f:
        for data in data_list:
            f.write(data.stem+"\n")

def make_dir(*dir_list):
    for dir_path in dir_list:
        Path(dir_path).mkdir(parents=True,exist_ok=True)


if __name__ == '__main__':

    pseusdo_path = "/nfs/neolix_data1/neolix_dataset/develop_dataset/pseudo_label_dataset/"
    label_path = pseusdo_path + "pesudo_labels"
    list_file = pseusdo_path + "ImageSets/train.txt"
    org_list_file = "/nfs/neolix_data1/neolix_dataset/develop_dataset/pseudo_label_dataset/ID_1026/ImageSets/train.txt"

    make_dir(label_path, Path(list_file).parent)
    if not Path(list_file).exists():
        print("generate list file: ", list_file)
        # generate_list_file(label_path, list_file, suffix="txt")
        generate_list_file(label_path, list_file, suffix="txt",org_list_file)

    cfg_file = "cfgs/dataset_configs/neolix_dataset.yaml"

    dataset_cfg = EasyDict(yaml.load(open(cfg_file)))
    ROOT_DIR = Path('/nfs/neolix_data1/neolix_dataset/develop_dataset/pseudo_label_dataset/')
    create_neolix_infos(
                                        dataset_cfg=dataset_cfg,
                                        class_names=['Vehicle', 'Large_vehicle', 'Pedestrian', 'Cyclist', 'Bicycle', 'Unknown_movable', 'Unknown_unmovable'],
                                        data_path=ROOT_DIR,
                                        save_path=ROOT_DIR
                                        )




