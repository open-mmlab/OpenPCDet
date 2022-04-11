#!/bin/bash

name='openpcd'

echo "Launching docker and status ..."

if docker ps -a --format '{{.Names}}' | grep -w $name &> /dev/null; then
	if docker ps -a --format '{{.Status}}' | egrep 'Exited' &> /dev/null; then
		echo "Container is already running. Attach to ${name}"
		docker start $name 	
		docker exec -it $name bash 
	elif docker ps -a --format '{{.Status}}' | egrep 'Created' &> /dev/null; then
		echo "Container is already created. Start and attach to ${name}"
		docker start $name 	
		docker exec -it $name bash
	elif docker ps -a --format '{{.Status}}' | egrep 'Up' &> /dev/null; then
		echo "Docker is already running"
		docker exec -it $name bash
	fi 
else

	echo "Starting ..."
	echo "docker run --name ${name} openpcdet-docker"
	docker run --name $name -it \
		-e DISPLAY=$DISPLAY \
		-e ROS_HOSTNAME=127.0.0.1 \
		-e ROS_MASTER_IP=https://${ROS_HOSTNAME}:11311 \
		-e ROS_IP=172.17.0.1 \
		-p ${ROS_HOSTNAME}:11311:11311 \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v /home/$USER/OpenPCDet/data/kitti:/OpenPCDet/data/kitti \
		-v /home/$USER/OpenPCDet/checkpoints:/OpenPCDet/checkpoints \
		-v /home/$USER/OpenPCDet/tools:/OpenPCDet/tools \
		-v /home/$USER/OpenPCDet/setup.py:/OpenPCDet/setup.py \
		--gpus all \
		--network=host \
		openpcdet-docker
fi


