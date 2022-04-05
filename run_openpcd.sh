#!/bin/bash

name='openpcd'

echo "Launching docker and status ..."

if docker ps -a --format '{{.Names}}' | grep -w $name &> /dev/null; then
	if docker ps -a --format '{{.Status}}' | egrep 'Exited' &> /dev/null; then
		echo "Container is already running. Attach to ${name}"
		docker start $name 	
		docker attach $name
	elif docker ps -a --format '{{.Status}}' | egrep 'Created' &> /dev/null; then
		echo "Container is already created. Start and attach to ${name}"
		docker start $name 	
		docker attach $name
	elif docker ps -a --format '{{.Status}}' | egrep 'Up' &> /dev/null; then
		echo "Docker is already running"
	fi 
else

	echo "Starting ..."
	echo "docker run --name ${name} openpcdet-docker"
	docker run --name $name -it \
		-e DISPLAY=$DISPLAY \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v /media/$USER/T7/Dataset/3D_data/kitti:/OpenPCDet/data/kitti \
		-v /home/$USER/OpenPCDet/checkpoints:/OpenPCDet/checkpoints \
		-v /home/$USER/OpenPCDet/tools:/OpenPCDet/tools \
		-v /home/$USER/OpenPCDet/setup.py:/OpenPCDet/setup.py \
		--rm --gpus all \
		openpcdet-docker
fi


