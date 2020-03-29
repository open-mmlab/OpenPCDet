#!/bin/sh

# Define data exchange folde
EXCHANGE="/home/bt"

docker run -it \
        --runtime=nvidia \
        --net=host \
        --privileged=true \
        --ipc=host \
        --volume="/tmp/.X11-unix:/tmp/.X11-unix:ro" \
        --env="DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
	--hostname="inside-DOCKER" \
	--name="pcdet" \
        -v $EXCHANGE:/root/exchange \
	pcdet-docker bash
