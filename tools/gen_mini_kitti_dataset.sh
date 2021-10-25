#!/bin/bash

NUM_SAMPLES=100

#/point_pillars/OpenPCDet/data/kitti
mkdir -p ../data/kitti/ImageSetsMini
pushd ../data/kitti/ImageSetsDefault
for f in *
do
	shuf -n $NUM_SAMPLES $f > ../ImageSetsMini/$f
done

cd ..
rm -f ImageSets
ln -s ImageSetsMini ImageSets
popd

pushd ..
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
cp data/kitti/*.pkl /point_pillars/kitti_pkl_backup/mini
popd
