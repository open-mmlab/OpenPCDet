#!/bin/bash

name='openpcd'

echo "Docker started"

if docker ps -a --format '{{.Names}}' | grep -w $name &> /dev/null; then
	if docker ps -a --format '{{.Status}}' | egrep 'Exited' &> /dev/null; then
		echo "Container is already running. Attach to ${name}"
		docker start $name 	
		docker attach $name
	fi 
else

	echo "Start to "docker run --name ${name}" openpcdet-docker"
	docker run --name $name  -it openpcdet-docker
fi
