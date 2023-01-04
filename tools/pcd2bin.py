#
# Module:       pcd2bin.py
# Description:  .pcd to .bin converter
#
# Author:       Yuseung Na (ys.na0220@gmail.com) origin link: https://github.com/Yuseung-Na/pcd2bin
# Version:      1.0
# Date:         January 19, 2021
#
# Author:       Kin ZHANG  (https://kin-zhang.github.io/)
# Version:      2.0
# Date:         2023-01-04

# Please run following to install the package: (TESTED on py37 py38), check issue: https://github.com/Yuseung-Na/pcd2bin/issues/8
# pip uninstall pypcd
# pip install --upgrade git+https://github.com/klintan/pypcd.git

# How to use:
# python3 pcd2bin.py --pcd_path /home/kin/test/data --bin_path /home/kin/test/data/bin

import numpy as np
import os
import argparse
from pypcd import pypcd
import csv
from tqdm import tqdm

def main():
    ## Add parser
    parser = argparse.ArgumentParser(description="Convert .pcd to .bin")
    parser.add_argument(
        "--pcd_path",
        help=".pcd file path.",
        type=str,
        default="/home/kin/test/data"
    )
    parser.add_argument(
        "--bin_path",
        help=".bin file path.",
        type=str,
        default="/home/kin/test/data"
    )
    parser.add_argument(
        "--file_name",
        help="File name.",
        type=str,
        default="file_name"
    )
    args = parser.parse_args()

    ## Find all pcd files
    pcd_files = []
    for (path, dir, files) in os.walk(args.pcd_path):
        for filename in files:
            # print(filename)
            ext = os.path.splitext(filename)[-1]
            if ext == '.pcd':
                pcd_files.append(path + "/" + filename)

    ## Sort pcd files by file name
    pcd_files.sort()   
    print("Finish to load point clouds!")

    ## Make bin_path directory
    try:
        if not (os.path.isdir(args.bin_path)):
            os.makedirs(os.path.join(args.bin_path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print ("Failed to create directory!!!!!")
            raise

    ## Generate csv meta file
    csv_file_path = os.path.join(args.bin_path, "meta.csv")
    csv_file = open(csv_file_path, "w")
    meta_file = csv.writer(
        csv_file, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
    )
    ## Write csv meta file header
    meta_file.writerow(
        [
            "pcd file name",
            "bin file name",
        ]
    )
    print("Finish to generate csv meta file")

    ## Converting Process
    print("Converting Start!")
    seq = 0
    for pcd_file in tqdm(pcd_files):
        ## Get pcd file
        pc = pypcd.PointCloud.from_path(pcd_file)
        
        pcd_file_name = pcd_file.split("/")[-1].split(".")[0]
        ## Generate bin file name
        bin_file_name = "{}.bin".format(pcd_file_name)
        bin_file_path = os.path.join(args.bin_path, bin_file_name)
        
        ## Get data from pcd (x, y, z, intensity, ring, time)
        np_x = (np.array(pc.pc_data['x'], dtype=np.float32)).astype(np.float32)
        np_y = (np.array(pc.pc_data['y'], dtype=np.float32)).astype(np.float32)
        np_z = (np.array(pc.pc_data['z'], dtype=np.float32)).astype(np.float32)
        np_i = np.zeros_like(np_x)
        if 'intensity' in pc.fields:
            np_i = (np.array(pc.pc_data['intensity'], dtype=np.float32)).astype(np.float32)/256
        elif 'Intensity' in pc.fields:
            np_i = (np.array(pc.pc_data['Intensity'], dtype=np.float32)).astype(np.float32)/256
        else:
            print(f"The pt fields: {pc.fields}. Please check whether there is an intensity field and add to codes")
        # np_r = (np.array(pc.pc_data['ring'], dtype=np.float32)).astype(np.float32)
        # np_t = (np.array(pc.pc_data['time'], dtype=np.float32)).astype(np.float32)

        ## Stack all data    
        points_32 = np.transpose(np.vstack((np_x, np_y, np_z, np_i)))

        ## Save bin file                                    
        points_32.tofile(bin_file_path)

        ## Write csv meta file
        meta_file.writerow(
            [os.path.split(pcd_file)[-1], bin_file_name]
        )

        seq = seq + 1
    
if __name__ == "__main__":
    main()