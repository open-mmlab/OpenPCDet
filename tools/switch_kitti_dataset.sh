#!/bin/bash

if [ -z $1 ]; then
	printf "Give cmd line arg, mini or default"
	exit
fi

if [ $1 == 'mini' ]; then
	cp /point_pillars/kitti_pkl_backup/mini/* ../data/kitti
elif [ $1 == 'default' ]; then
	cp /point_pillars/kitti_pkl_backup/default/* ../data/kitti
fi

pushd ../data/kitti
rm -f ImageSets
if [ $1 == 'mini' ]; then
	ln -s ImageSetsMini ImageSets
elif [ $1 == 'default' ]; then
	ln -s ImageSetsDefault ImageSets
fi
popd

